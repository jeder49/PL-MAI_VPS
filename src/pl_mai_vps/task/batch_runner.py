import abc
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
from numpy.typing import NDArray
from vllm.logprobs import Logprob

from pl_mai_vps.dataset.charades_dataset import get_or_download_charades_video_moments
from pl_mai_vps.dataset.dataset import Dataset, VideoMoment
from pl_mai_vps.task.vllm_model import VLLMModel
from pl_mai_vps.util.metrics import evaluate_moment_retrieval, MomentRetrievalMetrics
from pl_mai_vps.util.vllm_util import VLLMPrompt


class Evaluator[T](abc.ABC):
    def on_start(self):
        pass

    @abc.abstractmethod
    def on_end(self) -> T:
        raise NotImplementedError()

    @abc.abstractmethod
    def add_sample(self, sample: VideoMoment, predicted_windows: NDArray):
        raise NotImplementedError()


class MomentRetrievalEvaluator(Evaluator[MomentRetrievalMetrics]):
    def __init__(self):
        super().__init__()
        self.ground_truth_windows: list[NDArray] = []
        self.predicted_windows: list[NDArray] = []

    def on_start(self):
        self.ground_truth_windows = []
        self.predicted_windows = []

    def on_end(self) -> MomentRetrievalMetrics:
        return evaluate_moment_retrieval(self.predicted_windows, self.ground_truth_windows)

    def add_sample(self, sample: VideoMoment, predicted_windows: NDArray):
        ground_truth_windows = np.asarray(sample["windows"])
        # Expects shape [num_windows, 2] with [[start, end], [start,end], ...]
        # num_windows can be different between ground truth and predicted windows
        assert len(ground_truth_windows.shape) == 2 and ground_truth_windows.shape[1] == 2
        assert len(predicted_windows.shape) == 2 and predicted_windows.shape[1] == 2

        self.ground_truth_windows.append(ground_truth_windows)
        self.predicted_windows.append(predicted_windows)


class Processor[M](abc.ABC):
    def on_start(self):
        pass

    def on_end(self):
        pass

    @abc.abstractmethod
    def generate_prompts(self, dataset: Dataset, sample: VideoMoment) -> tuple[list[VLLMPrompt], M]:
        raise NotImplementedError()

    @abc.abstractmethod
    def process_results_to_windows(self, prompts: list[VLLMPrompt], results: list[tuple[str, list[dict[int, Logprob]]]],
                                   metadata: M) -> NDArray:
        raise NotImplementedError()


class OpenRequest:

    def __init__(self, sample: VideoMoment, prompts: list[VLLMPrompt], request_metadata: Any | None):
        super().__init__()
        self.sample = sample
        self.prompts = prompts
        self.results: list[tuple[str, list[dict[int, Logprob]]]] = []
        self.metadata = request_metadata

    def add_result(self, result: tuple[str, list[dict[int, Logprob]]]):
        self.results.append(result)
        return len(self.results) == len(self.prompts)


def run_moment_retrieval(model: VLLMModel, dataset: Dataset, processor: Processor, evaluator: Evaluator, batch_size=20):
    print()
    print("=" * 20)
    print(f"Running moment retrieval on dataset {dataset.name} with model {model.model_name}")
    print("=" * 20)
    print()

    # dataset.samples = dataset.samples[0:10]

    with tqdm.tqdm(desc="Running", total=len(dataset.samples)) as pbar:
        open_requests: dict[int, OpenRequest] = {}
        prompt_queue: deque[tuple[VLLMPrompt, int]] = deque()

        next_request_id = 0

        processor.on_start()
        evaluator.on_start()

        def finished_sample(sample: VideoMoment, result_windows: NDArray):
            evaluator.add_sample(sample, result_windows)
            pbar.update(1)

        def process_queue(drain: bool = False):
            while len(prompt_queue) > 0:
                if drain is False and len(prompt_queue) < batch_size:
                    return

                process_items: list[tuple[VLLMPrompt, int]] = []
                # Get batch (min needed when draining)
                for i in range(min(batch_size, len(prompt_queue))):
                    process_items.append(prompt_queue.popleft())

                batch_results = model.run_batch([x[0] for x in process_items])

                assert len(batch_results) == len(process_items)

                for (result, item) in zip(batch_results, process_items):
                    request_id = item[1]
                    request = open_requests[request_id]

                    close_request = request.add_result(result)

                    if close_request:
                        del open_requests[request_id]
                        result = processor.process_results_to_windows(request.prompts, request.results,
                                                                      request.metadata)
                        finished_sample(request.sample, result)

        for sample in dataset.samples:
            request_prompts, request_metadata = processor.generate_prompts(dataset, sample)
            open_requests[next_request_id] = OpenRequest(sample, request_prompts, request_metadata)

            for prompt in request_prompts:
                prompt_queue.append((prompt, next_request_id))

            next_request_id += 1
            process_queue()

        process_queue(drain=True)

        assert len(open_requests) == 0, f"Still have {len(open_requests)} open requests, which should not be the case"

    processor.on_end()
    return evaluator.on_end()

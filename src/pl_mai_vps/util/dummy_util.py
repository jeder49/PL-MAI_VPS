from typing import Any

import numpy as np
from vllm.logprobs import Logprob

from pl_mai_vps.dataset.dataset import VideoMoment, Dataset
from pl_mai_vps.task.batch_runner import Processor
from pl_mai_vps.util.vllm_util import VLLMPrompt
from numpy.typing import NDArray


class DummyProcessor(Processor[list[str]]):

    def generate_prompts(self, dataset: Dataset, sample: VideoMoment) -> tuple[list[VLLMPrompt], list[str]]:
        # Random number of prompts for testing purposes
        total_prompts = []
        metadata = []
        for i in range(np.random.randint(1, 20)):
            my_prompt = VLLMPrompt()
            my_prompt.add_system_text("You are a helpful chat assistant")
            my_prompt.add_user_text(f"Hello, please answer with '{sample["video_id"]}_{i}'")

            total_prompts.append(my_prompt)
            metadata.append(f"{sample["video_id"]}_{i}")

        return total_prompts, metadata

    def process_results_to_windows(self, prompts: list[VLLMPrompt], results: list[tuple[str, list[dict[int, Logprob]]]],
                                   metadata: list[str]) -> NDArray:
        return np.asarray([[0.0, 1.0], [5.0, 7.0]])


class DummyModel:
    def run_batch(self, prompts: list[list[Any]]) -> list[tuple[str, list[dict[int, Logprob]]]]:
        return [("dummy_output", []) for _ in prompts]

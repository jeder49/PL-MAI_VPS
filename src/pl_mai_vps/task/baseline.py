# Results on Charades-STA:
# {'r1': {'iou_thresholds': array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]), 'values': array([7.7688172 , 6.39784946, 4.65053763, 3.84408602, 2.60752688,
#        2.15053763, 1.3172043 , 0.67204301, 0.43010753, 0.21505376]), 'average': np.float64(3.0053763440860215), 'miou': np.float64(0.20341897505015988)}, 'map': {'iou_thresholds': array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]), 'values': array([7.7688172 , 6.39784946, 4.65053763, 3.84408602, 2.60752688,
#        2.15053763, 1.3172043 , 0.67204301, 0.43010753, 0.21505376]), 'average': np.float64(3.005376344086021)}}
# => R1@0.5= 7.77, R1@0.7 = 2.6

import numpy as np
import regex
from numpy._typing import NDArray
from vllm.logprobs import Logprob

from pl_mai_vps.dataset.dataset import Dataset, VideoMoment
from pl_mai_vps.task.batch_runner import Processor
from pl_mai_vps.util.vllm_util import VLLMPrompt


class BaselineProcessor(Processor[None]):

    def __init__(self, sample_amount: int = 20):
        super().__init__()

        # Handle string -> int conversion when called from command line
        if isinstance(sample_amount, str):
            sample_amount = int(sample_amount)

        self.sample_amount = sample_amount

        self.pattern = regex.compile(r"ANSWER: *\[ *(\d+(\.\d+)?) *, *(\d+(\.\d+)?) *\]")

    def generate_prompts(self, dataset: Dataset, sample: VideoMoment) -> tuple[list[VLLMPrompt], None]:
        # Following Chrono GPT-4o prompt

        prompt = VLLMPrompt()

        frame_seconds = np.linspace(0, sample["video_length"], self.sample_amount).tolist()

        frames = dataset.get_video_frames_multithreading(sample["video_id"], frame_seconds, num_threads=6)

        for timestamp, frame in zip(frame_seconds, frames):
            prompt.add_user_text(f"Frame at {timestamp:.2f} seconds:")
            prompt.add_user_image_from_cv2(frame)

        prompt.add_user_text(f"This video lasts {sample["video_length"]:.2f} seconds")

        prompt.add_user_text(f"Query: {sample["prompt"]}.")

        # prompt.add_user_text("""
        #     Given the video and the query, find the relevant windows. Think step by step. Reason about the
        #     events in the video and how they relate to the query. After your reasoning, output ‘ANSWER: <your
        #     answer>‘ in the format specified in the task prompt. Always provide a non-empty answer after your
        #     thoughts. If you think the event does not take place in the video, give your best guess, as otherwise
        #     the evaluation will be marked as incorrect. Never provide an empty list for <your answer>. The
        #     descriptions of moments are sometimes imprecise, so retrieve the closest moment. If you don’t see
        #     an event remotely similar to the description, guess what is the most likely moment given the context.
        #     For instance, for cutting onion this could be between the time we see that the scene takes place in the
        #     kitchen and the time we see the onions being boiled in the pan. The answer should be in the format of
        #     a list indicating the start and end of a window of moment, [start window, end window], for instance [0,
        #     1]. If you detect multiple windows for the same moment, choose the most relevant one. It’s important
        #     your final answer only contains one window. It is very important that the answer is in this format,
        #     otherwise the evaluation will fail.
        # """)

        # More optimized prompt
        prompt.add_user_text("""
            Find the time window where the queried event occurs in this video.

            First, analyze the video frames and identify when the event happens. Explain your reasoning.
            
            Your response must end with exactly this format as the final line: ANSWER: [start, end]
            
            If the exact event isn't visible, find the most likely time window based on context clues from the surrounding frames. Even if you have to make your best guess, you must always include the ANSWER line with a time window. Always provide exactly one time window.
        """)

        prompt.add_assistant_text("Let me think through this step by step:")

        return [prompt], None

    def process_results_to_windows(self, prompts: list[VLLMPrompt], results: list[tuple[str, list[dict[int, Logprob]]]],
                                   metadata: None) -> NDArray:
        assert len(results) == 1
        text_answer = results[0][0]

        # print(f"TOTAL ANSWER => <<<{text_answer}>>>")

        last_line = text_answer.split("\n")[-1].strip()

        full_match = self.pattern.fullmatch(last_line)

        if full_match is not None:
            # print(last_line, "=>", full_match.group(1), full_match.group(3))
            # print()
            return np.asarray([[float(full_match.group(1)), float(full_match.group(3))]])
        else:
            # No prediction
            return np.asarray(([[0.0, 0.0]]))

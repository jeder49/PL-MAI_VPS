from typing import Any

from vllm import LLM, RequestOutput
from vllm.logprobs import Logprob

from pl_mai_vps.util.vllm_util import VLLMPrompt


class VLLMModel:

    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", max_images_per_prompt: int = 60,
                 max_model_token_amount=6000, sampling_params: dict[str, Any] | None = None,
                 num_logprobs=20):
        super().__init__()
        self.model_name = model_name

        # Initialize the vLLM engine.
        self.llm_engine = LLM(
            model=model_name,
            max_model_len=max_model_token_amount,
            # No prefix caching, only make sense for chat conversations or static prompts
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": max_images_per_prompt}
        )

        self.sampling_params = self.llm_engine.get_default_sampling_params()

        self.sampling_params.logprobs = num_logprobs
        self.sampling_params.max_tokens = 512

        if sampling_params is not None:
            for (k, v) in sampling_params.items():
                self.sampling_params.__setattr__(k, v)

    def run_batch(self, prompts: list[VLLMPrompt]) -> list[tuple[str, list[dict[int, Logprob]]]]:
        batch_output: list[RequestOutput] = self.llm_engine.chat(
            [p.build() for p in prompts],
            sampling_params=self.sampling_params,
            use_tqdm=False
        )

        batch_results: list[tuple[str, list[dict[int, Logprob]]]] = []

        for i in range(len(prompts)):
            output = batch_output[i]
            text_outputs = output.outputs
            assert len(text_outputs) == 1

            logprobs = text_outputs[0].logprobs
            text = text_outputs[0].text
            if logprobs is not None:
                batch_results.append((text, logprobs))
            else:
                # Only happens if specifically deactivated logprobs
                batch_results.append((text, None))

        return batch_results

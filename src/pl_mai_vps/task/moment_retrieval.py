import json
import os
from pathlib import Path
from typing import Any

import click
import numpy as np

from pl_mai_vps.util.class_util import load_class_from_string
from pl_mai_vps.util.click_util import CLICK_JSON_DICT_TYPE
from pl_mai_vps.util.metrics import get_closest_threshold_idx


def get_datasets_dir():
    # Allow override via environment variable
    if datasets_path := os.getenv('DATASETS_DIR'):
        return Path(datasets_path)

    # Default to repo structure
    script_dir = Path(__file__).parent.parent.parent
    return script_dir.parent / "datasets"


@click.command()
@click.option("--processor", "processor_name", default="baseline.py/BaselineProcessor",
              help="<file_name>/<processor_class_name>")
@click.option("--processor-config", "processor_config", type=CLICK_JSON_DICT_TYPE,
              help='Processor config as JSON or key1=val1,key2=val2')
@click.option("--datasets-dir", "datasets_dir_parameter", default="auto",
              help="Where to download/retrieve the datasets from")
@click.option("--dataset", "dataset_name", default="charades-STA",
              help="Which dataset to use")
@click.option("--batch-size", default=16,
              help="Batch size for vLLM, minimum 1")
@click.option("--vllm-model-name", default="Qwen/Qwen2.5-VL-3B-Instruct",
              help='vLLM model name to use')
@click.option("--max-model-tokens", "max_model_token_amount", default=16384,
              help='How many tokens the model is allowed to handle. Increase if more frames are sampled.')
def run_moment_retrieval(processor_name: str, processor_config: dict[str, Any] | None, datasets_dir_parameter: str,
                         dataset_name: str, batch_size: int, vllm_model_name: str,
                         max_model_token_amount: int):
    processor_config = processor_config or {}

    datasets_dir = get_datasets_dir() if datasets_dir_parameter == "auto" else Path(datasets_dir_parameter)
    datasets_dir.mkdir(exist_ok=True)

    print("Using datasets directory:", datasets_dir)

    assert dataset_name == "charades-STA", "Currently only supports 'charades-STA' dataset!"
    ProcessorClass = load_class_from_string(processor_name)
    processor = ProcessorClass(**processor_config)

    # Locally load packages to speed up loading time of command line interface
    from pl_mai_vps.dataset.charades_dataset import get_or_download_charades_video_moments
    from pl_mai_vps.task.batch_runner import run_moment_retrieval, MomentRetrievalEvaluator
    from pl_mai_vps.task.vllm_model import VLLMModel
    from pl_mai_vps.util.metrics import MomentRetrievalMetrics

    # Download/Load Charades-STA
    train_dataset, validation_dataset, test_dataset = get_or_download_charades_video_moments(datasets_dir)

    # Actually run moment retrieval task on the test dataset
    metrics: MomentRetrievalMetrics = run_moment_retrieval(
        VLLMModel(vllm_model_name, max_model_token_amount=max_model_token_amount), test_dataset,
        processor,
        MomentRetrievalEvaluator(), batch_size=batch_size
    )

    print("=" * 20)
    print("Evaluation metrics")
    print("=" * 20)
    # print(a)
    print(json.dumps(metrics, sort_keys=True, indent=4))

    r05 = metrics["r1"]["values"][get_closest_threshold_idx(0.5, metrics["r1"]["iou_thresholds"])]
    r07 = metrics["r1"]["values"][get_closest_threshold_idx(0.7, metrics["r1"]["iou_thresholds"])]
    miou = metrics["r1"]["miou"]

    print()

    print(f"R1@0.5 = {r05:.3f}")
    print(f"R1@0.7 = {r07:.3f}")
    print(f"mIoU = {miou:.3f}")


if __name__ == '__main__':
    run_moment_retrieval()

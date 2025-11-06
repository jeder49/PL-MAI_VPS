import json
import time

import numpy as np

from pl_mai_vps.util.metrics import evaluate_moment_retrieval
from pl_mai_vps.util.original_metrics import eval_submission


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary into a single level with dot-separated keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compare_dicts(dict1, dict2):
    """
    Compare two nested dictionaries with the following criteria:
    a) Both dicts must have exactly the same keys (no extras or missing)
    b) Each float value must be equal when rounded to 2 decimal places

    Returns:
        tuple: (is_equal: bool, messages: list of str)
    """
    # Flatten both dictionaries
    flat1 = flatten_dict(dict1)
    flat2 = flatten_dict(dict2)

    messages = []
    is_equal = True

    # Check for missing or extra keys
    keys1 = set(flat1.keys())
    keys2 = set(flat2.keys())

    missing_in_dict2 = keys1 - keys2
    missing_in_dict1 = keys2 - keys1

    if missing_in_dict2:
        is_equal = False
        for key in missing_in_dict2:
            messages.append(f"Key '{key}' exists in dict1 but missing in dict2")

    if missing_in_dict1:
        is_equal = False
        for key in missing_in_dict1:
            messages.append(f"Key '{key}' exists in dict2 but missing in dict1")

    # Compare values for common keys
    common_keys = keys1 & keys2
    for key in common_keys:
        val1 = flat1[key]
        val2 = flat2[key]

        # Round to 2 decimal places for comparison
        rounded_val1 = round(val1, 2)
        rounded_val2 = round(val2, 2)

        if rounded_val1 != rounded_val2:
            is_equal = False
            messages.append(
                f"Value mismatch at '{key}': {val1} (rounded: {rounded_val1}) vs {val2} (rounded: {rounded_val2})")

    if is_equal and not messages:
        messages.append("Dictionaries are equal")

    return is_equal, messages


def convert_to_old_metrics(new_dict):
    r1_iou_thresholds = new_dict["r1"]["iou_thresholds"]
    map_iou_thresholds = new_dict["map"]["iou_thresholds"]

    map_thresholded = new_dict["map"]["values"]
    overall_map = new_dict["map"]["average"]

    r1_average = new_dict["r1"]["average"]
    r1_results = new_dict["r1"]["values"]

    miou = new_dict["r1"]["miou"]

    r1_iou_0_5_threshold_idx = np.argmin(np.abs(np.asarray(r1_iou_thresholds) - 0.5))
    r1_iou_0_7_threshold_idx = np.argmin(np.abs(np.asarray(r1_iou_thresholds) - 0.7))

    map_iou_0_5_threshold_idx = np.argmin(np.abs(np.asarray(map_iou_thresholds) - 0.5))
    map_iou_0_75_threshold_idx = np.argmin(np.abs(np.asarray(map_iou_thresholds) - 0.75))

    full_mr_map = {f"{t:.2f}".removesuffix("0"): r for (t, r) in zip(map_iou_thresholds, map_thresholded)}
    full_mr_map["average"] = overall_map

    return {
        "brief": {
            "MR-full-R1-avg": r1_average,
            "MR-full-R1@0.5": r1_results[r1_iou_0_5_threshold_idx],
            "MR-full-R1@0.7": r1_results[r1_iou_0_7_threshold_idx],
            "MR-full-invalid_pred_num": 0,
            "MR-full-mAP": overall_map,
            "MR-full-mAP@0.5": map_thresholded[map_iou_0_5_threshold_idx],
            "MR-full-mAP@0.75": map_thresholded[map_iou_0_75_threshold_idx],
            "MR-full-mIoU": miou
        },
        "full": {
            "MR-R1": {f"{t:.2f}".removesuffix("0"): r for (t, r) in zip(r1_iou_thresholds, r1_results)},
            "MR-R1-avg": r1_average,
            "MR-invalid_pred_num": 0,
            "MR-mAP": full_mr_map,
            "MR-mIoU": miou
        }
    }


def run_comparison(seed: int = 42, amount: int = 100):
    data = []

    """
     submission: list(dict), each dict is {
            qid: str,
            query: str,
            vid: str,
            pred_relevant_windows: list([st, ed]),
            pred_saliency_scores: list(float), len == #clips in video.
                i.e., each clip in the video will have a saliency score.
        }
        ground_truth: list(dict), each dict is     {
          "qid": 7803,
          "query": "Man in gray top walks from outside to inside.",
          "duration": 150,
          "vid": "RoripwjYFp8_360.0_510.0",
          "relevant_clip_ids": [13, 14, 15, 16, 17]
          "saliency_scores": [[4, 4, 2], [3, 4, 2], [2, 2, 3], [2, 2, 2], [0, 1, 3]]
               each sublist corresponds to one clip in relevant_clip_ids.
               The 3 elements in the sublist are scores from 3 different workers. The
               scores are in [0, 1, 2, 3, 4], meaning [Very Bad, ..., Good, Very Good]
        }
    """
    submission = []
    ground_truth = []

    rng = np.random.default_rng(seed)
    # Semi-random qids
    qids = np.arange(amount) * 10 + rng.integers(0, 10, size=[amount])

    def generate_random_windows(vid_duration: float, min_predictions=1, max_predictions=5) -> list[list[float]]:
        # Generate predictions with valid timestamps
        windows = rng.random(size=[rng.integers(min_predictions, max_predictions), 2]) * vid_duration
        # Sort windows to form valid start <= stop
        windows = np.sort(windows, axis=1).tolist()
        return windows

    for i in range(amount):
        qid = qids[i]
        vid_name = f"dummy_name_{qid}"
        query = f"Dummy query"

        # Random clips
        relevant_clip_ids = np.unique(rng.integers(0, amount, size=[rng.integers(1, 15)])).tolist()

        # Random saliency scores
        saliency_scores = rng.integers(0, 5, size=[len(relevant_clip_ids), 3]).tolist()

        # Random length from 1.0 - 20.0
        vid_duration = (rng.random() * 19.0 + 1.0)
        # Generate random windows
        pred_relevant_windows = generate_random_windows(vid_duration)
        relevant_windows = generate_random_windows(vid_duration)

        submission.append({
            "qid": qid,
            "vid": vid_name,
            "query": query,
            # We do not predict this for MR right now, so it's not needed
            # "pred_saliency_scores": ...,
            # Incorrectly needed by the original metrics
            "relevant_windows": relevant_windows,
            "pred_relevant_windows": pred_relevant_windows
        })

        ground_truth.append({
            "qid": qid,
            "vid": vid_name,
            "query": query,
            "relevant_clip_ids": relevant_clip_ids,
            "saliency_scores": saliency_scores,
            "relevant_windows": relevant_windows
        })

    # Shuffle them for realistic speed measuring
    rng.shuffle(submission)
    rng.shuffle(ground_truth)

    # print(submission)
    # print(ground_truth)

    old_start = time.perf_counter()
    old_evaluation_result = eval_submission(submission, ground_truth)
    print("Old evaluation took", time.perf_counter() - old_start, "seconds")

    # Sanity check that qids actually are the same when sorted
    assert sorted([x["qid"] for x in submission]) == sorted([x["qid"] for x in ground_truth])

    new_start = time.perf_counter()
    new_evaluation_result = convert_to_old_metrics(evaluate_moment_retrieval(
        [np.asarray(x["pred_relevant_windows"]) for x in sorted(submission, key=lambda x: x["qid"])],
        [np.asarray(x["relevant_windows"]) for x in sorted(ground_truth, key=lambda x: x["qid"])]
    ))
    print("New evaluation took", time.perf_counter() - new_start, "seconds")

    is_equal, messages = compare_dicts(old_evaluation_result, new_evaluation_result)

    # print(json.dumps(old_evaluation_result, sort_keys=True, indent=4))
    # print(json.dumps(new_evaluation_result, sort_keys=True, indent=4))

    assert is_equal, f"Error! {'\n'.join(messages)}"


if __name__ == '__main__':
    run_comparison(amount=10000, seed=200)

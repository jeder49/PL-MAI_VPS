from typing import Collection, TypedDict

import numpy as np
from numpy.typing import NDArray


class R1Metrics(TypedDict):
    iou_thresholds: list[float]
    values: list[float]
    average: float
    miou: float


class MapMetrics(TypedDict):
    iou_thresholds: list[float]
    values: list[float]
    average: float


class MomentRetrievalMetrics(TypedDict):
    r1: R1Metrics
    map: MapMetrics


def get_closest_threshold_idx(threshold_value: float, thresholds: list[float]):
    return np.argmin(np.abs(np.asarray(thresholds) - threshold_value))


def evaluate_moment_retrieval(predicted_windows: list[NDArray],
                              ground_truth_windows: list[NDArray]) -> MomentRetrievalMetrics:
    r1_iou_thresholds, r1_results, r1_average, miou = calculate_r1(predicted_windows, ground_truth_windows)

    map_iou_thresholds, map_thresholded, overall_map = calculate_mean_average_precision(predicted_windows,
                                                                                        ground_truth_windows)

    return {
        "r1": {
            "iou_thresholds": r1_iou_thresholds.tolist(),
            "values": r1_results.tolist(),
            "average": r1_average,
            "miou": miou
        },
        "map": {
            "iou_thresholds": map_iou_thresholds.tolist(),
            "values": map_thresholded.tolist(),
            "average": overall_map
        }
    }


def interpolated_precision_recall(precision, recall):
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns：
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        # noinspection PyTypeChecker
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap


def calculate_iou_scores(predicted_window: NDArray, ground_truth_windows: NDArray) -> NDArray:
    """
    Calculate IoU scores between one predicted 1D range and multiple ground truth 1D ranges.

    Args:
        predicted_window: 1D numpy array of shape (2,) representing [start, end]
        ground_truth_windows: 2D numpy array of shape (num_targets, 2) where each row is [start, end]

    Returns:
        1D numpy array of shape (num_targets,) containing IoU scores
    """
    # Extract start and end points
    pred_start, pred_end = predicted_window[0], predicted_window[1]
    gt_starts = ground_truth_windows[:, 0]
    gt_ends = ground_truth_windows[:, 1]

    # Calculate intersection bounds
    intersection_start = np.maximum(pred_start, gt_starts)
    intersection_end = np.minimum(pred_end, gt_ends)

    # Calculate intersection length (0 if no overlap)
    intersection_length = np.maximum(0, intersection_end - intersection_start)

    # Calculate union length
    pred_length = pred_end - pred_start
    gt_lengths = gt_ends - gt_starts
    union_length = pred_length + gt_lengths - intersection_length

    # Calculate IoU scores, handle division by zero
    iou_scores = np.where(union_length != 0, intersection_length / union_length, 0.0)

    return iou_scores


def calculate_best_iou_scores(predicted_windows: NDArray, ground_truth_windows: list[NDArray]):
    best_iou_scores = np.zeros(len(predicted_windows))

    for i in range(len(predicted_windows)):
        best_iou_scores[i] = np.max(calculate_iou_scores(predicted_windows[i, :], ground_truth_windows[i]))

    return best_iou_scores


def calculate_true_and_false_positives(predicted_windows: NDArray, ground_truth_windows: NDArray,
                                       iou_thresholds: NDArray):
    # Active is specific for each ground truth per threshold
    still_active_ground_truth_windows = np.ones([iou_thresholds.shape[0], ground_truth_windows.shape[0]], dtype=np.bool)
    # True = True positiv, False = False positive; for each threshold and predicted windows
    true_positives = np.zeros([iou_thresholds.shape[0], predicted_windows.shape[0]], dtype=np.bool)

    for i in range(len(predicted_windows)):
        # Get active scores for all thresholds
        active_iou_scores = calculate_iou_scores(
            predicted_windows[i],
            ground_truth_windows)[np.newaxis, :] * still_active_ground_truth_windows
        # Greedily mark best score as used ground truth window per threshold
        best_ground_truth_idx = np.argmax(active_iou_scores, axis=1)
        best_ground_truth_values = np.max(active_iou_scores, axis=1)

        # Deactivate used ground truth windows
        deactivate = best_ground_truth_values >= iou_thresholds
        deactivation_mask = np.zeros_like(still_active_ground_truth_windows)
        deactivation_mask[np.arange(iou_thresholds.shape[0]), best_ground_truth_idx] = deactivate

        # Mark as true positive accordingly
        true_positives[:, i] = deactivate

        still_active_ground_truth_windows = np.logical_and(
            still_active_ground_truth_windows,
            np.logical_not(deactivation_mask)
        )

    return true_positives


def calculate_precision_and_recall(true_and_false_positives: NDArray, total_items: int):
    # Expects shape [num_thresholds, num_predictions]
    assert len(true_and_false_positives.shape) == 2

    num_predictions = true_and_false_positives.shape[1]

    tp_cumsum = true_and_false_positives.cumsum(axis=1)

    prediction_sum = np.arange(num_predictions) + 1

    recall = tp_cumsum / total_items
    precision = tp_cumsum / prediction_sum

    return precision, recall


def calculate_mean_average_precision(predicted_windows: list[NDArray], ground_truth_windows: list[NDArray],
                                     iou_thresholds: Collection[float] = np.linspace(0.5, 0.95, 10)) -> tuple[
    NDArray, NDArray, float]:
    average_precisions = np.zeros([len(iou_thresholds), len(predicted_windows)])

    for i in range(len(predicted_windows)):
        # Calculate precision and recall for all thresholds and rank ranges
        precision, recall = calculate_precision_and_recall(
            calculate_true_and_false_positives(predicted_windows[i], ground_truth_windows[i],
                                               np.asarray(iou_thresholds)),
            ground_truth_windows[i].shape[0]
        )

        # Mean precision for each threshold, considering all rank ranges
        for threshold_idx in range(len(iou_thresholds)):
            average_precisions[threshold_idx, i] = interpolated_precision_recall(
                precision[threshold_idx, :],
                recall[threshold_idx, :]
            )

    threshold_mean_average_precisions = np.mean(average_precisions, axis=1)
    overall_mean_average_precision = np.mean(threshold_mean_average_precisions)
    return np.asarray(iou_thresholds), threshold_mean_average_precisions * 100.0, overall_mean_average_precision * 100.0


def calculate_r1(predicted_windows: list[NDArray], ground_truth_windows: list[NDArray],
                 iou_thresholds: Collection[float] = np.linspace(0.5, 0.95, 10)) -> tuple[
    NDArray, NDArray, float, float]:
    num_samples = len(predicted_windows)
    assert num_samples == len(
        ground_truth_windows), f"Inconsistent predicted_windows and ground_truth_windows lengths: {len(predicted_windows)} vs {len(ground_truth_windows)}"

    # Select only the first predicted windows
    predicted_windows = np.asarray([x[0, :] for x in predicted_windows])

    # Calculate best ground truth window for each predicted window and the iou score of it
    iou_scores = calculate_best_iou_scores(predicted_windows, ground_truth_windows)

    # Calculate different thresholds memory-efficient in a loop, rather than one big numpy call
    r1_results = np.asarray([np.mean(iou_scores >= iou_threshold) * 100.0 for iou_threshold in iou_thresholds])

    r1_average = np.mean(r1_results)
    miou = np.mean(iou_scores)

    return np.asarray(iou_thresholds), r1_results, r1_average, miou

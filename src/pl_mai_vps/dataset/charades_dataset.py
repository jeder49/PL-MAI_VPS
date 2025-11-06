import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from pl_mai_vps.dataset.dataset import VideoMoment, Dataset
from pl_mai_vps.util.file_util import download_with_verification, json_to_csv_in_memory, save_data_to_file, \
    process_zip_from_url, csv_bytes_to_dict, read_json_file, extract_directory_from_zip
from pl_mai_vps.util.video_util import write_frame_to_file

CHARADES_480P_VIDEOS_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
CHARADES_ANNOTATIONS_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip"
CHARADES_STA_TRAIN = "https://raw.githubusercontent.com/MCG-NJU/MMN/refs/heads/main/dataset/Charades_STA/charades_train.json"
CHARADES_STA_TEST = "https://raw.githubusercontent.com/MCG-NJU/MMN/refs/heads/main/dataset/Charades_STA/charades_test.json"


def create_or_replace_dir(dir_path: Path, delete_existing: bool):
    if not delete_existing:
        dir_path.mkdir(exist_ok=True)
        return

    # Delete existing
    if dir_path.exists():
        shutil.rmtree(dir_path)

    dir_path.mkdir(exist_ok=False)


def download_videos(videos_dir: Path):
    zip_path = videos_dir.parent / "all_videos.zip"
    download_with_verification(CHARADES_480P_VIDEOS_URL, zip_path)

    extract_directory_from_zip(zip_path, "Charades_v1_480/", videos_dir)

    # Remove now redundant zip file
    os.remove(zip_path)


def download_train_and_test_jsons(train_json: Path, test_json: Path):
    train_csv = "Charades/Charades_v1_train.csv"
    test_csv = "Charades/Charades_v1_test.csv"
    extracted = process_zip_from_url(
        CHARADES_ANNOTATIONS_URL,
        [train_csv, test_csv]
    )

    train_csv_data = extracted[train_csv]
    test_csv_data = extracted[test_csv]

    save_data_to_file(json.dumps(csv_bytes_to_dict(train_csv_data, "id")), train_json)
    save_data_to_file(json.dumps(csv_bytes_to_dict(test_csv_data, "id")), test_json)


def download_sta_train_and_test_jsons(train_json: Path, test_json: Path):
    download_with_verification(CHARADES_STA_TRAIN, train_json)
    download_with_verification(CHARADES_STA_TEST, test_json)


def get_or_download_charades_sta_if_necessary(datasets_dir: Path, delete_existing: bool = False):
    assert datasets_dir.is_dir()

    charades_dir = datasets_dir / "charades"
    create_or_replace_dir(charades_dir, delete_existing)

    videos_dir = charades_dir / "videos"
    create_or_replace_dir(videos_dir, delete_existing)

    # Check if all videos are extracted successfully
    if len(os.listdir(videos_dir)) != 9848:
        download_videos(videos_dir)

    train_json = charades_dir / "train.json"
    test_json = charades_dir / "test.json"

    if not train_json.exists() or not test_json.exists():
        download_train_and_test_jsons(train_json, test_json)

    charades_sta_dir = datasets_dir / "charades_sta"
    create_or_replace_dir(charades_sta_dir, delete_existing)

    sta_train_json = charades_sta_dir / "train.json"
    sta_test_json = charades_sta_dir / "test.json"

    if not sta_train_json.exists() or not sta_test_json.exists():
        download_sta_train_and_test_jsons(sta_train_json, sta_test_json)

    videos_processed_dir = charades_dir / "videos_processed"
    if videos_processed_dir.exists():
        print("Warning: Using processed dir, results might vary")
        videos_dir = videos_processed_dir

    return videos_dir, train_json, test_json, sta_train_json, sta_test_json


def get_video_moments(
        sta_json: dict, charades_json: Any, video_ids: list[str] | None = None
) -> list[VideoMoment]:
    prompt_list = []

    for (key, row) in sta_json.items():

        if video_ids is not None and key not in video_ids:
            continue

        duration = row["duration"]
        timestamps = row["timestamps"]
        sentences = row["sentences"]

        assert len(timestamps) == len(sentences)

        zipped_prompts = zip(timestamps, sentences)

        # zipped_prompts = list(set([(tuple(x[0]), tuple(x[1])) for x in zip(timestamps, sentences)]))
        # if len(timestamps) != len(zipped_prompts):
        #     print("Reduced from", zip(timestamps, sentences), "to", zipped_prompts)

        assert duration == float(charades_json[key]["length"]), f"{duration} vs {charades_json[key]["length"]}"

        for (window, prompt) in zipped_prompts:
            prompt_list.append({
                "video_id": key,
                "video_length": duration,
                "windows": [[window[0], window[1]]],
                "prompt": prompt
            })

    return prompt_list


def get_or_download_charades_video_moments(datasets_dir: Path, delete_existing: bool = False) -> tuple[
    Dataset, Dataset, Dataset]:
    videos_dir, train_json, test_json, sta_train_json, sta_test_json = get_or_download_charades_sta_if_necessary(
        datasets_dir, delete_existing)

    charades_chrono_dir = datasets_dir / "charades_chrono"
    create_or_replace_dir(charades_chrono_dir, delete_existing)

    train_moments_json = charades_chrono_dir / "train.json"
    validation_moments_json = charades_chrono_dir / "validation.json"
    test_moments_json = charades_chrono_dir / "test.json"

    if train_moments_json.exists() and validation_moments_json.exists() and test_moments_json.exists():
        return Dataset("charades-STA-train", read_json_file(train_moments_json), videos_dir, video_file_suffix=".mp4"), \
            Dataset("charades-STA-validation", read_json_file(validation_moments_json), videos_dir,
                    video_file_suffix=".mp4"), \
            Dataset("charades-STA-test", read_json_file(test_moments_json), videos_dir, video_file_suffix=".mp4")

    train_df = read_json_file(train_json)
    test_df = read_json_file(test_json)

    # get all unique ids
    train_ids = list(train_df.keys())
    # print(len(train_ids))

    # Same split as per https://github.com/sudo-Boris/mr-Blip/blob/main/mr_BLIP_data/data_preprocess.ipynb
    # randomly select 800 ids for validation
    random.seed(42)
    random.shuffle(train_ids)

    val_ids = train_ids[:800]
    train_ids = train_ids[800:]

    # print(len(val_ids), len(train_ids))

    train_moments = get_video_moments(read_json_file(sta_train_json), train_df, train_ids)
    validation_moments = get_video_moments(read_json_file(sta_train_json), train_df, val_ids)
    test_moments = get_video_moments(read_json_file(sta_test_json), test_df)

    save_data_to_file(json.dumps(train_moments), train_moments_json)
    save_data_to_file(json.dumps(validation_moments), validation_moments_json)
    save_data_to_file(json.dumps(test_moments), test_moments_json)

    # print(len(train_moments), len(validation_moments), len(test_moments))

    return Dataset("charades-STA-train", train_moments, videos_dir, video_file_suffix=".mp4"), \
        Dataset("charades-STA-validation", validation_moments, videos_dir, video_file_suffix=".mp4"), \
        Dataset("charades-STA-test", test_moments, videos_dir, video_file_suffix=".mp4")


if __name__ == '__main__':
    assert len(sys.argv) >= 2, "Expected a directory as argument!"
    datasets_dir = Path(sys.argv[1])
    train_dataset, validation_dataset, test_dataset = get_or_download_charades_video_moments(
        datasets_dir,
        delete_existing=False
    )

    # temp_dir = datasets_dir / "temp"
    # create_or_replace_dir(temp_dir, delete_existing=True)
    #
    # first_sample = train_dataset.samples[1]
    # print(first_sample)
    # timestamps = np.linspace(0, first_sample["video_length"], 60).tolist()
    # video_frames = train_dataset.get_video_frames(first_sample["video_id"], timestamps)
    #
    # for idx in range(len(timestamps)):
    #     write_frame_to_file(video_frames[idx], temp_dir / f"frame_{idx}.jpg")

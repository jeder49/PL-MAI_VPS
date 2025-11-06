import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tqdm
from numpy.typing import NDArray
import cv2

from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


def extract_frames_from_video_by_time(video_path: Path, timestamps: list[float], output_dir: Path | None = None) -> \
        list[
            NDArray]:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    # Sort timestamps to minimize seeking
    sorted_timestamps = sorted(timestamps)

    for timestamp in sorted_timestamps:
        # Use milliseconds for more accurate seeking
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()

        if ret:
            frames.append(frame)
            if output_dir:
                cv2.imwrite(str(output_dir / f"frame_{timestamp}s.jpg"), frame)

    cap.release()
    return frames


def write_frame_to_file(frame_data: NDArray, frame_path: Path):
    cv2.imwrite(str(frame_path), frame_data)


def extract_frames_multiprocessing(video_path: Path, timestamps: list[float], output_dir: Path | None = None,
                                   num_processes: int = 4) -> list[NDArray]:
    """Extract frames using multiprocessing with continuous chunks"""

    # Sort timestamps to ensure continuity
    sorted_timestamps = sorted(timestamps)

    # Create continuous chunks
    chunk_size = len(sorted_timestamps) // num_processes
    remainder = len(sorted_timestamps) % num_processes

    chunks = []
    start_idx = 0

    for i in range(num_processes):
        # Add one extra item to first 'remainder' chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        if current_chunk_size > 0:
            end_idx = start_idx + current_chunk_size
            chunks.append(sorted_timestamps[start_idx:end_idx])
            start_idx = end_idx

    # Create arguments for each process
    args = [(video_path, chunk, output_dir) for chunk in chunks if chunk]

    # Execute in parallel
    with Pool(processes=len(args)) as pool:
        results = pool.starmap(extract_frames_from_video_by_time, args)

    # Flatten results and maintain original order
    all_frames = []
    for result in results:
        all_frames.extend(result)

    # Reorder frames to match original timestamp order
    timestamp_to_frame = {}
    chunk_idx = 0
    for i, chunk in enumerate(chunks):
        for j, timestamp in enumerate(chunk):
            if i < len(results) and j < len(results[i]):
                timestamp_to_frame[timestamp] = results[i][j]

    # Return frames in original timestamp order
    ordered_frames = []
    for timestamp in timestamps:
        if timestamp in timestamp_to_frame:
            ordered_frames.append(timestamp_to_frame[timestamp])

    return ordered_frames


def extract_frames_multithreading(video_path: Path, timestamps: list[float], output_dir: Path | None = None,
                                  num_threads: int = 4) -> list[NDArray]:
    """Extract frames using multithreading with continuous chunks"""

    # Sort timestamps to ensure continuity
    sorted_timestamps = sorted(timestamps)

    # Create continuous chunks
    chunk_size = len(sorted_timestamps) // num_threads
    remainder = len(sorted_timestamps) % num_threads

    chunks = []
    start_idx = 0

    for i in range(num_threads):
        # Add one extra item to first 'remainder' chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        if current_chunk_size > 0:
            end_idx = start_idx + current_chunk_size
            chunks.append(sorted_timestamps[start_idx:end_idx])
            start_idx = end_idx

    # Execute in parallel using ThreadPoolExecutor
    chunk_results = []
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = []
        for chunk in chunks:
            if chunk:  # Only submit non-empty chunks
                future = executor.submit(extract_frames_from_video_by_time, video_path, chunk, output_dir)
                futures.append((future, chunk))

        # Collect results
        for future, chunk in futures:
            result = future.result()
            chunk_results.append((chunk, result))

    # Create timestamp to frame mapping
    timestamp_to_frame = {}
    for chunk, frames in chunk_results:
        for timestamp, frame in zip(chunk, frames):
            timestamp_to_frame[timestamp] = frame

    # Return frames in original timestamp order
    ordered_frames = []
    for timestamp in timestamps:
        if timestamp in timestamp_to_frame:
            ordered_frames.append(timestamp_to_frame[timestamp])

    return ordered_frames


if __name__ == '__main__':
    # Speed test for different setups
    assert len(sys.argv) >= 2, "Expected a directory as argument!"
    video_dir = Path(sys.argv[1])

    all_videos = [video_dir / x for x in os.listdir(video_dir)]

    total_frames_count = 0
    total_area_times_frames_count = 0

    total_times = {"normal": 0.0, "multithreading": 0.0, "multiprocessing": 0.0}

    # all_videos = all_videos[:10]
    all_videos = sorted(all_videos)[:100]

    for video_path in tqdm.tqdm(all_videos):
        cap = cv2.VideoCapture(str(video_path))
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        total_time = num_frames / fps

        # Try different approaches, track time and sum it up completely
        start = time.perf_counter()
        extract_frames_from_video_by_time(video_path, np.linspace(0, total_time, 60).tolist())
        total_times["normal"] = total_times["normal"] + (time.perf_counter() - start)
        start = time.perf_counter()
        extract_frames_multithreading(video_path, np.linspace(0, total_time, 60).tolist(), num_threads=8)
        total_times["multithreading"] = total_times["multithreading"] + (time.perf_counter() - start)
        start = time.perf_counter()
        extract_frames_multiprocessing(video_path, np.linspace(0, total_time, 60).tolist(), num_processes=8)
        total_times["multiprocessing"] = total_times["multiprocessing"] + (time.perf_counter() - start)

        total_frames_count += int(num_frames)
        total_area = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_area_times_frames_count += int(total_area * num_frames)

    # 4 => Averages: {'normal': '0.71', 'multithreading': '0.19', 'multiprocessing': '0.33'}

    # 4 => Averages: {'normal': '1.08', 'multithreading': '0.81', 'multiprocessing': '0.96'}

    # 4 => Averages: {'normal': '3.52', 'multithreading': '2.64', 'multiprocessing': '2.79'}
    # 8 => Averages: {'normal': '3.76', 'multithreading': '2.60', 'multiprocessing': '2.72'}
    print("Averages:", {k: f"{v / len(total_times):.2f}" for (k, v) in total_times.items()})

    # Standard quality:
    # Averages: {'normal': '35.44', 'multithreading': '22.52', 'multiprocessing': '24.31'}
    # Fastest, but has low quality:
    # ffmpeg -i input.mp4 -c:v mjpeg -q:v 5 -an output.avi
    # Slower, but better quality:
    # for file in videos/*.mp4; do ffmpeg -i "$file" -c:v libx264  -g 1 -crf 15 -preset ultrafast -an -y "videos_processed/$(basename "$file")"; done
    # ffmpeg -i input.mp4 -c:v libx264 -g 1 -crf 15 -preset ultrafast -an output.mp4
    # Averages: {'normal': '8.72', 'multithreading': '5.11', 'multiprocessing': '6.38'}

    print(total_frames_count)
    print(total_area_times_frames_count)
    # 7712599

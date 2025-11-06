from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from numpy.typing import NDArray

from pl_mai_vps.util.video_util import extract_frames_from_video_by_time, extract_frames_multithreading, \
    extract_frames_multiprocessing


class VideoMoment(TypedDict):
    video_id: str
    video_length: float
    windows: list[list[float]]
    prompt: str


@dataclass
class Dataset:
    name: str
    samples: list[VideoMoment]
    video_directory: Path
    video_file_prefix: str = ""
    video_file_suffix: str = ""

    def get_video_frames(self, video_id: str, timestamps: list[float]) -> list[NDArray]:
        return extract_frames_from_video_by_time(
            self.video_directory / (self.video_file_prefix + video_id + self.video_file_suffix),
            timestamps
        )

    def get_video_frames_multithreading(self, video_id: str, timestamps: list[float], num_threads: int = 4) -> list[
        NDArray]:
        return extract_frames_multithreading(
            self.video_directory / (self.video_file_prefix + video_id + self.video_file_suffix),
            timestamps, num_threads=num_threads
        )

    def get_video_frames_multiprocessing(self, video_id: str, timestamps: list[float], num_processes: int = 4) -> list[
        NDArray]:
        return extract_frames_multiprocessing(
            self.video_directory / (self.video_file_prefix + video_id + self.video_file_suffix),
            timestamps,
            num_processes=num_processes
        )

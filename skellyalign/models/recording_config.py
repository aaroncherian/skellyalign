from dataclasses import dataclass
from typing import Union, Tuple, Optional
from pathlib import Path


@dataclass
class RecordingConfig:
    path_to_recording: Union[str, Path]
    path_to_freemocap_output_data: Union[str, Path]
    path_to_qualisys_output_data: Union[str, Path]
    freemocap_markers: list
    qualisys_markers: list
    markers_for_alignment: list
    frame_for_alignment: int


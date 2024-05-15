from typing import Union, Tuple, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError

class RecordingConfig(BaseModel):
    recording_name: str
    path_to_recording: Union[str, Path]
    path_to_freemocap_output_data: Union[str, Path]
    path_to_qualisys_output_data: Union[str, Path]
    qualisys_marker_list: List[str]
    markers_to_compare_list: List[str]
    frames_to_sample: int = Field(20, gt=0, description="Number of frames to sample in each RANSAC iteration")
    max_iterations: int = Field(20, gt=0, description="Maximum number of RANSAC iterations")
    inlier_threshold: float = Field(50, gt=0, description="Inlier threshold for RANSAC")
    frame_range: Optional[Tuple[int, Optional[int]]] = None

    @field_validator('path_to_recording', 'path_to_freemocap_output_data', 'path_to_qualisys_output_data', mode='before')
    def validate_paths(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        return path
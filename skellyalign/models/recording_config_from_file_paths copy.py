from typing import Union, Tuple, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator
import numpy as np

class RecordingConfig(BaseModel):
    path_to_recording: Union[str, Path]
    path_to_freemocap_output_data: Union[str, Path]
    path_to_qualisys_output_data: Union[str, Path]
    freemocap_markers: List[str]
    qualisys_markers: List[str]
    markers_for_alignment: List[str]
    frames_to_sample: int = Field(20, gt=0, description="Number of frames to sample in each RANSAC iteration")
    max_iterations: int = Field(20, gt=0, description="Maximum number of RANSAC iterations")
    inlier_threshold: float = Field(50, gt=0, description="Inlier threshold for RANSAC")

    @field_validator('path_to_recording', 'path_to_freemocap_output_data', 'path_to_qualisys_output_data', mode='after')
    def validate_paths(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        return path

    @model_validator(mode='after')
    def validate_marker_presence(cls, values):
        freemocap_markers = values.freemocap_markers
        qualisys_markers = values.qualisys_markers
        markers_for_alignment = values.markers_for_alignment
        missing_in_freemocap = set(markers_for_alignment) - set(freemocap_markers)
        missing_in_qualisys = set(markers_for_alignment) - set(qualisys_markers)
        if missing_in_freemocap:
            raise ValueError(f"Markers for alignment not found in FreeMoCap markers: {missing_in_freemocap}")
        if missing_in_qualisys:
            raise ValueError(f"Markers for alignment not found in Qualisys markers: {missing_in_qualisys}")
        return values
    
    @model_validator(mode='after')
    def validate_marker_count(cls, values):
        freemocap_path = values.path_to_freemocap_output_data
        qualisys_path = values.path_to_qualisys_output_data
        freemocap_markers = values.freemocap_markers
        qualisys_markers = values.qualisys_markers

        try:
            freemocap_data = np.load(freemocap_path)
        except Exception as e:
            raise ValueError(f"Failed to load FreeMoCap data from {freemocap_path}: {e}")

        try:
            qualisys_data = np.load(qualisys_path)
        except Exception as e:
            raise ValueError(f"Failed to load Qualisys data from {qualisys_path}: {e}")

        if freemocap_data.shape[1] != len(freemocap_markers):
            raise ValueError(f"Number of markers in FreeMoCap data ({freemocap_data.shape[1]}) does not match the number of markers in the FreeMoCap marker list ({len(freemocap_markers)})")

        if qualisys_data.shape[1] != len(qualisys_markers):
            raise ValueError(f"Number of markers in Qualisys data ({qualisys_data.shape[1]}) does not match the number of markers in the Qualisys marker list ({len(qualisys_markers)})")

        return values
from typing import Callable, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator
import numpy as np
from skellymodels.skeleton_models.skeleton import Skeleton

# class SpatialAlignmentConfig(BaseModel):
#     freemocap_skeleton: Skeleton
#     qualisys_skeleton: Skeleton
#     markers_for_alignment: List[str]
#     frames_to_sample: int = Field(20, gt=0, description="Number of frames to sample in each RANSAC iteration")
#     max_iterations: int = Field(20, gt=0, description="Maximum number of RANSAC iterations")
#     inlier_threshold: float = Field(50, gt=0, description="Inlier threshold for RANSAC")

#     @model_validator(mode='after')
#     def validate_marker_presence(cls, values):
#         freemocap_markers = values.freemocap_skeleton.marker_names
#         qualisys_markers = values.qualisys_skeleton.marker_names
#         markers_for_alignment = values.markers_for_alignment
#         missing_in_freemocap = set(markers_for_alignment) - set(freemocap_markers)
#         missing_in_qualisys = set(markers_for_alignment) - set(qualisys_markers)
#         if missing_in_freemocap:
#             raise ValueError(f"These markers for alignment were not found in FreeMoCap markers: {missing_in_freemocap}")
#         if missing_in_qualisys:
#             raise ValueError(f"These markers for alignment were not found in Qualisys markers: {missing_in_qualisys}")
#         return values
    

class SpatialAlignmentConfig(BaseModel):
    path_to_freemocap_recording_folder:Path
    path_to_freemocap_output_data: Path
    freemocap_skeleton_function: Callable[[], Skeleton]
    path_to_qualisys_output_data: Path
    qualisys_skeleton_function: Callable[[], Skeleton]
    markers_for_alignment: List[str]
    frames_to_sample: int = Field(20, gt=0, description="Number of frames to sample in each RANSAC iteration")
    max_iterations: int = Field(20, gt=0, description="Maximum number of RANSAC iterations")
    inlier_threshold: float = Field(50, gt=0, description="Inlier threshold for RANSAC")

    @model_validator(mode='after')
    def check_paths_and_load_data(cls, values):
        path_to_freemocap_output_data = values.path_to_freemocap_output_data
        path_to_qualisys_output_data = values.path_to_qualisys_output_data

        # Check if paths exist
        if not path_to_freemocap_output_data.exists():
            raise ValueError(f"Path does not exist: {path_to_freemocap_output_data}")
        if not path_to_qualisys_output_data.exists():
            raise ValueError(f"Path does not exist: {path_to_qualisys_output_data}")

        # Try loading the numpy data
        try:
            np.load(path_to_freemocap_output_data)
        except Exception as e:
            raise ValueError(f"Failed to load numpy data from {path_to_freemocap_output_data}: {e}")

        try:
            np.load(path_to_qualisys_output_data)
        except Exception as e:
            raise ValueError(f"Failed to load numpy data from {path_to_qualisys_output_data}: {e}")

        return values
from typing import Union, Tuple, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator
import numpy as np
from skellymodels.skeleton_models.skeleton import Skeleton

class SpatialAlignmentConfig(BaseModel):
    freemocap_skeleton: Skeleton
    qualisys_skeleton: Skeleton
    markers_for_alignment: List[str]
    frames_to_sample: int = Field(20, gt=0, description="Number of frames to sample in each RANSAC iteration")
    max_iterations: int = Field(20, gt=0, description="Maximum number of RANSAC iterations")
    inlier_threshold: float = Field(50, gt=0, description="Inlier threshold for RANSAC")

    @model_validator(mode='after')
    def validate_marker_presence(cls, values):
        freemocap_markers = values.freemocap_skeleton.marker_names
        qualisys_markers = values.qualisys_skeleton.marker_names
        markers_for_alignment = values.markers_for_alignment
        missing_in_freemocap = set(markers_for_alignment) - set(freemocap_markers)
        missing_in_qualisys = set(markers_for_alignment) - set(qualisys_markers)
        if missing_in_freemocap:
            raise ValueError(f"These markers for alignment were not found in FreeMoCap markers: {missing_in_freemocap}")
        if missing_in_qualisys:
            raise ValueError(f"These markers for alignment were not found in Qualisys markers: {missing_in_qualisys}")
        return values
    

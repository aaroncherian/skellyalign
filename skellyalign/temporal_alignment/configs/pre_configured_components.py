from dataclasses import dataclass
from .recording_config import Component

@dataclass
class MediaPipeComponent(Component):
    """Pre-configured MediaPipe component"""
    def __init__(self):
        super().__init__(
            name='mediapipe',
            files={'body': 'mediapipe_body_3d_xyz.npy'},
            base_folder=None
        )
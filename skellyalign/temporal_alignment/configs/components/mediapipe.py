from skellyalign.temporal_alignment.configs.recording_config import Component
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo

mediapipe_metadata = {
    'landmark_names':MediapipeModelInfo.landmark_names
}
class MediaPipeComponent(Component):
    """Pre-configured MediaPipe component"""
    def __init__(self):
        super().__init__(
            name='mediapipe',
            files={'body': 'mediapipe_body_3d_xyz.npy'},
            base_folder=None
        )
        
        # Add MediaPipe-specific metadata
        mediapipe_metadata = {
            'landmark_names': MediapipeModelInfo.landmark_names
        }
        for key, value in mediapipe_metadata.items():
            self.metadata.add(key, value)

        
        
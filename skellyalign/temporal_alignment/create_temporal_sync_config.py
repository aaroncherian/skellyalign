from .configs.recording_config import Recording
from .configs.pre_configured_components import MediaPipeComponent
from typing import Union, Optional, List
from pathlib import Path


def create_temporal_sync_config(recording_folder_path: Union[str, Path]) -> Recording:
    recording = Recording(
        recording_folder_path=recording_folder_path,
    )
    
    recording.add_prepared_component(
        MediaPipeComponent()
    )

    recording.add_component(
    name = 'freemocap_timestamps',
    files = {
        'timestamps': 'unix_synced_timestamps.csv'
    },
)

    recording.add_component(
        name = 'qualisys_exported_markers',
        files = { 
            'markers': 'qualisys_exported_markers.tsv'
        },
        base_folder = 'component_qualisys_original'
    )

    return recording


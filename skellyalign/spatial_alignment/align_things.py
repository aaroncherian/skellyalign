from skellymodels.experimental.model_redo.managers.human import Human
from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo

from pathlib import Path 

path_to_recording = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2')

path_to_freemocap_original_data = path_to_recording / 'output_data'/ 'mediapipe_body_3d_xyz.npy'
path_to_qualisys_synced_data = path_to_recording/ 'output_data' / 'component_qualisys_synced'/  
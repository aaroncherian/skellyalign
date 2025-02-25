from skellyalign.temporal_alignment.create_temporal_sync_config import create_temporal_sync_config
from skellyalign.temporal_alignment.configs.recording_config import Recording


from typing import List
from skellyalign.temporal_alignment.temporal_synchronizer import TemporalSyncManager


def validate_required_metadata(component_name: str, required_keys: List[str]):
    component = recording_config.components[component_name]
    for key in required_keys:
        if not component.metadata.has(key):
            raise ValueError(f"Missing required metadata '{key}' for component {component_name}")

def run_temporal_synchronization(recording_config: Recording):

    # temporal_synchronizer = TemporalSynchronizer(recording_config)
    # temporal_synchronizer.align_data()
    return TemporalSyncManager(recording_config).run()


if __name__ in {"__main__", "__mp_main__"}:
    from skellyalign.temporal_alignment.create_temporal_sync_config import create_temporal_sync_config
    from skellyalign.temporal_alignment.marker_sets.full_body_joint_center_weights import joint_center_weights
    from pathlib import Path

    def setup_recording_config(path: str) -> Recording:
        config = create_temporal_sync_config(path)
        config.components['qualisys_exported_markers'].metadata.add(
            'joint_center_weights', 
            joint_center_weights
        )
        config.components['qualisys_exported_markers'].metadata.add(
            'joint_center_names', 
            list(joint_center_weights.keys())
        )
        return config
    
    recording_folder_path= r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2"
    
    recording_config = setup_recording_config(recording_folder_path)



    marker_data_synced, joint_center_data_synced = run_temporal_synchronization(recording_config)

    folder_to_save_qualisys_data = Path(recording_config.output_path/'component_qualisys_synced')
    folder_to_save_qualisys_data.mkdir(parents = True, exist_ok=True)

    marker_data_synced.to_csv(folder_to_save_qualisys_data/'qualisys_markers_synced.csv', index = False)

    joint_center_data_synced.to_csv(folder_to_save_qualisys_data/'qualisys_joint_centers_synced.csv', index=False)

    import pandas as pd
    import numpy as np
    import re

def convert_csv_to_npy(csv_file_path, npy_file_path):
    """
    Converts a CSV file containing motion capture data into an .npy file with 
    the format [frame, marker, dimension].

    Parameters:
    csv_file_path (str): Path to the input CSV file.
    npy_file_path (str): Path to save the output .npy file.

    Returns:
    tuple: Shape of the saved numpy array and the file path.
    """
    import numpy as np
    import pandas as pd

    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Extract relevant data (excluding Time, Frame, and Unix timestamps columns)
    marker_columns = [col for col in df.columns if col not in ["Time", "Frame", "unix_timestamps"]]

    # Determine the number of frames and markers
    num_frames = len(df)
    num_markers = len(marker_columns) // 3  # Each marker has x, y, z coordinates

    # Reshape data into [frame, marker, dimension] format
    data_reshaped = df[marker_columns].to_numpy().reshape(num_frames, num_markers, 3)

    # Save to an .npy file
    np.save(npy_file_path, data_reshaped)

    return data_reshaped.shape, npy_file_path



    
convert_csv_to_npy(csv_file_path=folder_to_save_qualisys_data/'qualisys_joint_centers_synced.csv', 
                             npy_file_path = folder_to_save_qualisys_data/'qualisys_joint_centers_synced.npy')


from skellyalign.temporal_alignment.create_temporal_sync_config import create_temporal_sync_config

from skellyalign.temporal_alignment.qualisys_data_processing.process_qualisys_tsv import load_qualisys_data, get_unix_start_time, extract_just_marker_data_from_dataframe, extract_marker_names, create_marker_array, calculate_joint_centers, create_joint_center_df, create_and_insert_unix_timestamp_column
from skellyalign.temporal_alignment.qualisys_data_processing.joint_center_weights.full_body_joint_center_weights import joint_center_weights
import pandas as pd

def get_header_length(file_path):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if line.startswith('TRAJECTORY_TYPES'):  # Detect the specific line for the header's end
                print(f'Header found, skipping {i+1} rows')
                return i + 1  # Data starts right after the marker names row
    print('Header not found')
    return 0  # Default if no header is found

def run_temporal_synchronization(recording_folder_path: str):
    recording_config = create_temporal_sync_config(recording_folder_path)

    qualisys_marker_tsv_path = recording_config.get_component_file_path('qualisys_exported_markers', 'markers')
    qualisys_marker_data_and_timestamps = load_qualisys_data(qualisys_marker_tsv_path)
    unix_start_time = get_unix_start_time(qualisys_marker_tsv_path)

    qualisys_marker_trajectories = extract_just_marker_data_from_dataframe(qualisys_marker_data_and_timestamps)

    marker_names = extract_marker_names(qualisys_marker_trajectories)

    qualisys_marker_trajectories_array= create_marker_array(qualisys_marker_trajectories)

    qualisys_joint_centers = calculate_joint_centers(
        marker_array=qualisys_marker_trajectories_array,
        marker_names=marker_names,
        joint_center_weights=joint_center_weights
    )

    joint_center_dataframe = create_joint_center_df(
        joint_center_array=qualisys_joint_centers,
        joint_names=list(joint_center_weights.keys()),
        dataframe_with_timestamps= qualisys_marker_data_and_timestamps
    )

    joint_center_dataframe_with_unix_timestamps = create_and_insert_unix_timestamp_column(
        df=joint_center_dataframe,
        unix_start_timestamp=unix_start_time
    )

    f = 2



if __name__ == '__main__':
    recording_folder_path= r"D:\2024-10-30_treadmill_pilot\processed_data\sesh_2024-10-30_15_45_14_mdn_gait_7_exposure"
    recording_config = run_temporal_synchronization(recording_folder_path)

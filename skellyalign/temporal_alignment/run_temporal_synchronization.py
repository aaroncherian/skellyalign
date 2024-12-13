from skellyalign.temporal_alignment.create_temporal_sync_config import create_temporal_sync_config

from skellyalign.temporal_alignment.qualisys_data_processing.process_qualisys_tsv import TSVProcessor, JointCenterCalculator, get_unix_start_time
from skellyalign.temporal_alignment.qualisys_data_processing.joint_center_weights.full_body_joint_center_weights import joint_center_weights
import pandas as pd


def run_temporal_synchronization(recording_folder_path: str):
    recording_config = create_temporal_sync_config(recording_folder_path)
    qualisys_marker_tsv_path = recording_config.get_component_file_path('qualisys_exported_markers', 'markers')
    unix_start_time = get_unix_start_time(qualisys_marker_tsv_path)

    
    qualisys_tsv_processor = TSVProcessor(qualisys_marker_tsv_path)
    qualisys_marker_and_timestamp_dataframe = qualisys_tsv_processor.get_qualisys_marker_tsv_data()

    joint_center_calculator = JointCenterCalculator(
        marker_and_timestamp_df=qualisys_marker_and_timestamp_dataframe,
        joint_center_weights=joint_center_weights
    )
    joint_center_calculator.calculate_joint_centers()
    joint_center_dataframe = joint_center_calculator.create_dataframe_with_unix_timestamps(unix_start_time=unix_start_time)

    
    f = 2
    





if __name__ == '__main__':
    recording_folder_path= r"D:\2024-10-30_treadmill_pilot\processed_data\sesh_2024-10-30_15_45_14_mdn_gait_7_exposure"
    recording_config = run_temporal_synchronization(recording_folder_path)

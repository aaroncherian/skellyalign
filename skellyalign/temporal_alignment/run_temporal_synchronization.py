from skellyalign.temporal_alignment.create_temporal_sync_config import create_temporal_sync_config

from skellyalign.temporal_alignment.qualisys_data_processing.process_qualisys_tsv import TSVProcessor, JointCenterCalculator, get_unix_start_time
from skellyalign.temporal_alignment.qualisys_data_processing.joint_center_weights.full_body_joint_center_weights import joint_center_weights

from skellyalign.temporal_alignment.freemocap_data_processing import create_freemocap_unix_timestamps
from skellyalign.temporal_alignment.qualisys_data_processing.resample_qualisys_data import QualisysResampler
from skellyalign.temporal_alignment.run_skellyforge_rotation import run_skellyforge_rotation
from skellyalign.temporal_alignment.configs.temporal_alignment_config import LagCorrectionSystemComponent, LagCorrector

from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo



import numpy as np

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

    freemocap_timestamps, framerate = create_freemocap_unix_timestamps(csv_path=recording_config.get_component_file_path('freemocap_timestamps', 'timestamps'))

    resampler = QualisysResampler(joint_center_dataframe, freemocap_timestamps, list(joint_center_weights.keys()))
    qualisys_joint_centers_resampled_rotated = resampler.rotated_resampled_marker_array

    freemocap_joint_centers = np.load(recording_config.get_component_file_path('mediapipe', 'body'))

    freemocap_joint_centers_rotated = run_skellyforge_rotation(freemocap_joint_centers, MediapipeModelInfo.landmark_names)
    
    freemocap_component = LagCorrectionSystemComponent(joint_center_array=freemocap_joint_centers_rotated, list_of_joint_center_names=MediapipeModelInfo.landmark_names)
    qualisys_component = LagCorrectionSystemComponent(joint_center_array=qualisys_joint_centers_resampled_rotated, list_of_joint_center_names=list(joint_center_weights.keys()))
    
    lag_corrector =LagCorrector(freemocap_component=freemocap_component, qualisys_component=qualisys_component, framerate=framerate)
    lag_corrector.run()

    median_lag = lag_corrector.median_lag
    print(f"Median lag: {median_lag}")

    lag_in_seconds = lag_corrector.get_lag_in_seconds()

    lag_corrected_dataframe = joint_center_calculator.create_dataframe_with_unix_timestamps(unix_start_time=unix_start_time, lag_in_seconds=lag_in_seconds)
    resampler_two = QualisysResampler(lag_corrected_dataframe, freemocap_timestamps, list(joint_center_weights.keys()))
    qualisys_joint_centers_resampled_rotated_two = resampler_two.rotated_resampled_marker_array

    qualisys_component_two = LagCorrectionSystemComponent(joint_center_array=qualisys_joint_centers_resampled_rotated_two, list_of_joint_center_names=list(joint_center_weights.keys()))
    lag_corrector_two = LagCorrector(freemocap_component=freemocap_component, qualisys_component=qualisys_component_two, framerate=framerate)

    lag_corrector_two.run()
    median_lag_two = lag_corrector_two.median_lag
    print(f"Median lag after second correction: {median_lag_two}")
    f = 2
    





if __name__ == '__main__':
    recording_folder_path= r"D:\2024-10-30_treadmill_pilot\processed_data\sesh_2024-10-30_15_45_14_mdn_gait_7_exposure"
    recording_config = run_temporal_synchronization(recording_folder_path)

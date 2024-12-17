from dataclasses import dataclass

from skellyalign.temporal_alignment.configs.recording_config import Recording

from typing import List

from skellyalign.temporal_alignment.qualisys_data_processing.process_qualisys_tsv import TSVProcessor, JointCenterCalculator, get_unix_start_time

from skellyalign.temporal_alignment.freemocap_data_processing import create_freemocap_unix_timestamps

import numpy as np
from skellyalign.temporal_alignment.run_skellyforge_rotation import run_skellyforge_rotation
from skellyalign.temporal_alignment.configs.temporal_alignment_config import LagCorrectionSystemComponent, LagCorrector

from skellyalign.temporal_alignment.qualisys_data_processing.resample_qualisys_data import QualisysResampler

@dataclass
class TemporalSynchronizer:

    recording_config: Recording

    def __post_init__(self):
        self._validate_required_metadata('qualisys_exported_markers', ['joint_center_weights', 'joint_center_names'])
        self._initialize_components()

    def _initialize_components(self):
        
        self.qualisys_joint_center_weights = self.recording_config.components['qualisys_exported_markers'].metadata.get('joint_center_weights')
        self.qualisys_joint_center_names = self.recording_config.components['qualisys_exported_markers'].metadata.get('joint_center_names')
        self.freemocap_joint_center_names = self.recording_config.components['mediapipe'].metadata.get('landmark_names')


        self.qualisys_marker_tsv_path = self.recording_config.get_component_file_path(
            'qualisys_exported_markers', 'markers')
        self.timestamps_path = self.recording_config.get_component_file_path(
            'freemocap_timestamps', 'timestamps')
        self.freemocap_joint_center_data_path = self.recording_config.get_component_file_path(
            'mediapipe', 'body')
        
        self.unix_start_time = get_unix_start_time(self.qualisys_marker_tsv_path)

    def _validate_required_metadata(self, component_name: str, required_keys: List[str]):
        component = self.recording_config.components[component_name]
        for key in required_keys:
            if not component.metadata.has(key):
                raise ValueError(f"Missing required metadata '{key}' for component {component_name}")


    def _process_qualisys_data(self, 
                               marker_tsv_path,
                               joint_center_weights):

        marker_and_time_data = TSVProcessor(marker_tsv_path).clean_up_qualisys_tsv()

        self.joint_center_calculator = JointCenterCalculator(
            marker_and_timestamp_df=marker_and_time_data,
            joint_center_weights=joint_center_weights
        )

        self.joint_center_calculator.calculate_joint_centers()

  
        
    def _get_timestamps_and_framerate(self, timestamps_path):
        self.freemocap_timestamps, self.framerate = create_freemocap_unix_timestamps(csv_path=timestamps_path)
        

    def _process_freemocap_data(self,joint_center_data_path, joint_center_names):
        joint_center_data = np.load(joint_center_data_path)

        joint_centers_rotated = run_skellyforge_rotation(
            raw_skeleton_data=joint_center_data, 
            landmark_names=joint_center_names)
        
        self.freemocap_component = LagCorrectionSystemComponent(
            joint_center_array=joint_centers_rotated, 
            list_of_joint_center_names=joint_center_names)
        


    def _create_qualisys_component(self, lag_in_seconds:float = 0) -> LagCorrectionSystemComponent:
        joint_centers_df = self.joint_center_calculator.create_dataframe_with_unix_timestamps(
            unix_start_time=self.unix_start_time, 
            lag_in_seconds=lag_in_seconds)
        
        resampler = QualisysResampler(joint_centers_df, 
                                        self.freemocap_timestamps, 
                                        self.qualisys_joint_center_names)
        
        return LagCorrectionSystemComponent(
        joint_center_array=resampler.rotated_resampled_marker_array,
        list_of_joint_center_names=self.qualisys_joint_center_names
    )

    def _calculate_lag(self, qualisys_component: LagCorrectionSystemComponent):
        lag_corrector = LagCorrector(
            freemocap_component=self.freemocap_component, 
            qualisys_component=qualisys_component, 
            framerate=self.framerate)
        
        lag_corrector.run()
        print('Median lag:', lag_corrector.median_lag)
        print('Lag in seconds:', lag_corrector.get_lag_in_seconds())
        return lag_corrector.get_lag_in_seconds()

    def align_data(self):

        
        self._get_timestamps_and_framerate(
            timestamps_path=self.timestamps_path
        )

        self._process_freemocap_data(
            joint_center_data_path=self.freemocap_joint_center_data_path, 
            joint_center_names=self.freemocap_joint_center_names
        )

        self._process_qualisys_data(
            marker_tsv_path=self.qualisys_marker_tsv_path,
            joint_center_weights=self.qualisys_joint_center_weights  
        )

        qualisys_component = self._create_qualisys_component(lag_in_seconds=0)
        initial_lag = self._calculate_lag(qualisys_component)

        corrected_qualisys_component = self._create_qualisys_component(lag_in_seconds=initial_lag)
        final_lag = self._calculate_lag(corrected_qualisys_component)

        print('Initial lag:', initial_lag)
        print('Final lag:', final_lag)


 

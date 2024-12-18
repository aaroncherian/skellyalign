from skellyalign.temporal_alignment.create_temporal_sync_config import create_temporal_sync_config

from skellyalign.temporal_alignment.qualisys_data_processing.process_qualisys_tsv import TSVProcessor, JointCenterCalculator, get_unix_start_time
from skellyalign.temporal_alignment.freemocap_data_processing import create_freemocap_unix_timestamps
from skellyalign.temporal_alignment.qualisys_data_processing.resample_qualisys_data import QualisysResampler
from skellyalign.temporal_alignment.run_skellyforge_rotation import run_skellyforge_rotation
from skellyalign.temporal_alignment.configs.temporal_alignment_config import LagCorrectionSystemComponent, LagCorrector

from skellyalign.temporal_alignment.configs.recording_config import Recording

import numpy as np

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
    TemporalSyncManager(recording_config).run()


if __name__ == '__main__':
    from skellyalign.temporal_alignment.create_temporal_sync_config import create_temporal_sync_config
    from skellyalign.temporal_alignment.qualisys_data_processing.joint_center_weights.full_body_joint_center_weights import joint_center_weights

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
    
    recording_folder_path= r"D:\2024-10-30_treadmill_pilot\processed_data\sesh_2024-10-30_15_45_14_mdn_gait_7_exposure"
    
    recording_config = setup_recording_config(recording_folder_path)



    run_temporal_synchronization(recording_config)

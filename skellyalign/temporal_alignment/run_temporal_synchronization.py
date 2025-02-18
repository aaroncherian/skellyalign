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


if __name__ == '__main__':
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
    
    recording_folder_path= r"D:\mdn_data\sesh_2023-05-17_13_48_44_MDN_treadmill_2"
    
    recording_config = setup_recording_config(recording_folder_path)



    marker_data_synced = run_temporal_synchronization(recording_config)
    marker_data_synced.to_csv(Path(recording_config.output_path/'component_qualisys_original'/'marker_data_synced.csv'), index=False)

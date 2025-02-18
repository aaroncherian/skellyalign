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
    TemporalSyncManager(recording_config).run()


if __name__ == '__main__':
    from skellyalign.temporal_alignment.create_temporal_sync_config import create_temporal_sync_config
    from skellyalign.temporal_alignment.marker_sets.full_body_joint_center_weights import joint_center_weights

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
    
    recording_folder_path= r"D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_44_19_P01_WalkRun_Trial1"
    
    recording_config = setup_recording_config(recording_folder_path)



    run_temporal_synchronization(recording_config)

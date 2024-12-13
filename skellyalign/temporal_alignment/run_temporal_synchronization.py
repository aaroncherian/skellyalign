from skellyalign.temporal_alignment.create_temporal_sync_config import create_temporal_sync_config
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
    header_length = get_header_length(qualisys_marker_tsv_path)
    qualisys_marker_trajectories = pd.read_csv(qualisys_marker_tsv_path, delimiter='\t', skiprows=header_length)

    f = 2



if __name__ == '__main__':
    recording_folder_path= r"D:\2024-10-30_treadmill_pilot\processed_data\sesh_2024-10-30_15_45_14_mdn_gait_7_exposure"
    recording_config = run_temporal_synchronization(recording_folder_path)

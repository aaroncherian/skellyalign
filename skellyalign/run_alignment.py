from skellyalign.models.recording_config import RecordingConfig
from skellyalign.utilities.data_handlers import DataLoader, DataProcessor
import numpy as np

def main(recording_config:RecordingConfig):
    print(recording_config)

    # freemocap_data = np.load(recording_config.path_to_freemocap_output_data)
    qualisys_data = np.load(recording_config.path_to_qualisys_output_data)

    freemocap_dataloader = DataLoader(path=recording_config.path_to_freemocap_output_data)
    freemocap_data_processor = DataProcessor(data=freemocap_dataloader.data_3d, marker_list=recording_config.freemocap_markers)

    f = 2   




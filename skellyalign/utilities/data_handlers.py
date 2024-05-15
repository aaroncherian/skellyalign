import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Optional, List
from skellyalign.models.recording_config import RecordingConfig

class DataLoader:
    def __init__(self, path: Union[str, Path] = None, data_array: Optional[np.ndarray] = None):
        self.path = Path(path) if path else None
        self._data_array = data_array
        self._data = self.load_data()  # Automatically load data upon initialization

    def load_data(self):
        if self.path:
            print(f"Loading data from {self.path}.")
            try:
                return np.load(self.path)
            except Exception as e:
                raise ValueError(f"Failed to load data from {self.path}: {e}")
        elif self._data_array is not None:
            print(f"Loading data from provided array.")
            return self._data_array
        else:
            raise ValueError("Either path or data_array must be provided.")

    @property
    def data_3d(self):
        return self._data
    

class DataProcessor:
    def __init__(self, data: np.ndarray, marker_list: List[str]):
        self._data = data
        self.marker_list = marker_list

        self._extracted_data = self.extract_markers()

    def extract_markers(self, markers_to_extract: List[str]):
        indices = [self.marker_list.index(marker) for marker in markers_to_extract]
        extracted_data = self._data[:, indices, :]
        return extracted_data

    def convert_to_dataframe(self, extracted_data: np.ndarray):
        num_frames, num_markers, _ = extracted_data.shape
        reshaped_data = extracted_data.reshape(num_frames, num_markers * 3)
        columns = [f"{marker}_{axis}" for marker in self.marker_list for axis in ['x', 'y', 'z']]
        dataframe = pd.DataFrame(reshaped_data, columns=columns)
        return dataframe
    
    @property
    def data_3d(self):
        return self._data
    
    @property
    def extracted_data_3d(self):
        return self._extracted_data
    


# def main(recording_config: RecordingConfig):
#     # Load FreeMoCap data
#     freemocap_loader = DataLoader(path=recording_config.path_to_freemocap_output_data)
#     freemocap_data = freemocap_loader.load_data()

#     # Load Qualisys data
#     qualisys_loader = DataLoader(path=recording_config.path_to_qualisys_output_data)
#     qualisys_data = qualisys_loader.load_data()

#     # Process FreeMoCap data
#     freemocap_processor = DataProcessor(data=freemocap_data, marker_list=recording_config.freemocap_markers)
#     freemocap_extracted = freemocap_processor.extract_markers(recording_config.markers_for_alignment)
#     freemocap_dataframe = freemocap_processor.convert_to_dataframe(freemocap_extracted)

#     # Process Qualisys data
#     qualisys_processor = DataProcessor(data=qualisys_data, marker_list=recording_config.qualisys_markers)
#     qualisys_extracted = qualisys_processor.extract_markers(recording_config.markers_for_alignment)
#     qualisys_dataframe = qualisys_processor.convert_to_dataframe(qualisys_extracted)

#     # Add further processing or analysis here
#     # ...

#     return {
#         "freemocap_data": freemocap_data,
#         "freemocap_extracted": freemocap_extracted,
#         "freemocap_dataframe": freemocap_dataframe,
#         "qualisys_data": qualisys_data,
#         "qualisys_extracted": qualisys_extracted,
#         "qualisys_dataframe": qualisys_dataframe
#     }


# if __name__ = '__main__':

import numpy as np
import pandas as pd 

import config

# Define more generic constants
DATA_3D_ARRAY = 'data_3d_array'
EXTRACTED_3D_ARRAY = 'extracted_3d_array'
DATAFRAME_OF_3D_DATA = 'dataframe_of_3d_data'

class DataFrameBuilder:
    def __init__(self, path_to_data, marker_list):
        self.path_to_data = path_to_data
        self.marker_list = marker_list
        self.data = {}

        self.data_3d_array = None
        self.extracted_3d_array = None
        self.dataframe_of_3d_data = None
    
    def load_data(self):
        self.data_3d_array = np.load(self.path_to_data)
        return self
    
    def extract_common_markers(self):
        if self.data_3d_array is None:
            raise ValueError(f"{DATA_3D_ARRAY} is None. You must run load_data() first.")
            
        self.extracted_3d_array = self._extract_specific_markers(
            data_marker_dimension=self.data_3d_array,
            list_of_markers=self.marker_list,
            markers_to_extract=config.markers_to_extract)
        return self

    def convert_to_dataframe(self):
        if self.extracted_3d_array is None:
            raise ValueError(f"{EXTRACTED_3D_ARRAY} is None. You must run extract_common_markers() first.")
            
        self.dataframe_of_3d_data = self._convert_3d_array_to_dataframe(
            data_3d_array=self.extracted_3d_array,
            data_marker_list=config.markers_to_extract)
        return self

    def build(self):

        return {
            DATA_3D_ARRAY: self.data_3d_array,
            EXTRACTED_3D_ARRAY: self.extracted_3d_array,
            DATAFRAME_OF_3D_DATA: self.dataframe_of_3d_data
        }

    def _extract_specific_markers(self,data_marker_dimension:np.ndarray, list_of_markers:list, markers_to_extract:list):
        """
        Extracts specific markers for a frame of a 3D data array based on the given indices and markers to extract.

        Parameters:
        - data (numpy.ndarray): The 3D data array containing all markers. Shape should be (num_markers, 3).
        - indices (list): The list of marker names corresponding to the columns in the data array.
        - markers_to_extract (list): The list of marker names to extract.

        Returns:
        - extracted_data (numpy.ndarray): A new 3D data array containing only the extracted markers. 
        Shape will be (num_frames, num_extracted_markers, 3).
        """
        # Identify the column indices that correspond to the markers to extract
        col_indices = [list_of_markers.index(marker) for marker in markers_to_extract if marker in list_of_markers]
        
        # Extract the relevant columns from the data array
        extracted_data = data_marker_dimension[:,col_indices, :]

        return extracted_data
    
    def _convert_3d_array_to_dataframe(self, data_3d_array:np.ndarray, data_marker_list:list):
        """
        Convert the FreeMoCap data from a numpy array to a pandas DataFrame.

        Parameters:
        - data_3d_array (numpy.ndarray): The 3d data array. Shape should be (num_frames, num_markers, 3).
        - data_marker_list (list): List of marker names 

        Returns:
        - data_frame_marker_dim_dataframe (pandas.DataFrame): DataFrame containing FreeMoCap data with columns ['Frame', 'Marker', 'X', 'Y', 'Z'].

        """
        num_frames = data_3d_array.shape[0]
        num_markers = data_3d_array.shape[1]

        frame_list = []
        marker_list = []
        x_list = []
        y_list = []
        z_list = []

        for frame in range(num_frames):
            for marker in range(num_markers):
                frame_list.append(frame)
                marker_list.append(data_marker_list[marker])
                x_list.append(data_3d_array[frame, marker, 0])
                y_list.append(data_3d_array[frame, marker, 1])
                z_list.append(data_3d_array[frame, marker, 2])

        data_frame_marker_dim_dataframe = pd.DataFrame({
            'frame': frame_list,
            'marker': marker_list,
            'x': x_list,
            'y': y_list,
            'z': z_list
        })

        return data_frame_marker_dim_dataframe

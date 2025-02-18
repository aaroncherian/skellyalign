from skellyalign.temporal_alignment.configs.recording_config import Recording
from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from skellyalign.temporal_alignment.utils.rotation import run_skellyforge_rotation
from datetime import datetime

class QualisysMarkerData:
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def load_tsv(self) -> pd.DataFrame:
        """Load the TSV file, skipping the header."""
        header_length = self._get_header_length()
        self.data = pd.read_csv(
            self.file_path,
            delimiter='\t',
            skiprows=header_length
        )
        return self.data

    def _get_header_length(self) -> int:
        """Determine the length of the header in the TSV file."""
        with self.file_path.open('r') as file:
            for i, line in enumerate(file):
                if line.startswith('TRAJECTORY_TYPES'):
                    return i + 1
        raise ValueError("Header not found in the TSV file.")
    
    def _extract_marker_data(self) -> pd.DataFrame:
        """Extract only marker data columns."""

        columns_of_interest = self.data.columns[
            ~self.data.columns.str.contains(r'^(?:Frame|Time|unix_timestamps|Unnamed)', regex=True)
        ]
        return self.data[columns_of_interest]
    
    @property
    def marker_names(self) -> List[str]:
        marker_columns = self._extract_marker_data().columns
        return list(dict.fromkeys(col.split()[0] for col in marker_columns))
    
    @property
    def marker_array(self) -> np.ndarray:
        marker_data = self._extract_marker_data()
        num_frames = len(marker_data)
        num_markers = int(len(marker_data.columns) / 3)
        return marker_data.to_numpy().reshape(num_frames, num_markers, 3)
    
    @property 
    def time_and_frame_columns(self) -> pd.DataFrame:
        return self.data[['Time', 'Frame']]

    @property
    def unix_start_time(self) -> float:
        """Extract and convert the Qualisys starting timestamp to Unix time."""
        with open(self.file_path, 'r') as file:
            for line in file:
                if line.startswith('TIME_STAMP'):
                    timestamp_str = line.strip().split('\t')[1]
                    datetime_obj = datetime.strptime(timestamp_str, '%Y-%m-%d, %H:%M:%S.%f')
                    return datetime_obj.timestamp()
        raise ValueError(f"No TIME_STAMP found in file: {self.file_path}")
    
    def as_dataframe_with_unix_timestamps(self, lag_seconds: float = 0) -> pd.DataFrame:
        df = self.as_dataframe()
        df['unix_timestamps'] = df['Time'] + self.marker_data.unix_start_time + lag_seconds
        return df

class QualisysJointCenterData:

    def __init__(self, marker_data_holder:QualisysMarkerData, weights:Dict):
        self.marker_data = marker_data_holder
        self.weights = weights
        self.joint_names = list(weights.keys())
        self.joint_centers = self._calculate_joint_centers(
            marker_data_array=marker_data_holder.marker_array,
            marker_names=marker_data_holder.marker_names,
            joint_center_weights=weights
        )


    def _calculate_joint_centers(self, marker_data_array:np.ndarray, marker_names:List, joint_center_weights:Dict):
        """
        Optimized calculation of joint centers for Qualisys data with 3D weights.

        Parameters:
            marker_array (np.ndarray): Shape (num_frames, num_markers, 3), 3D marker data.
            joint_center_weights (dict): Weights for each joint as {joint_name: {marker_name: [weight_x, weight_y, weight_z]}}.
            marker_names (list): List of marker names corresponding to marker_array.

        Result:
            np.ndarray: Joint centers with shape (num_frames, num_joints, 3).
        """
        print('Calculating joint centers...')
        num_frames, num_markers, _ = marker_data_array.shape
        num_joints = len(joint_center_weights)

        marker_to_index = {marker: i for i, marker in enumerate(marker_names)}

        weights_matrix = np.zeros((num_joints, num_markers, 3))
        for j_idx, (joint, markers_weights) in enumerate(joint_center_weights.items()):
            for marker, weight in markers_weights.items():
                marker_idx = marker_to_index[marker]
                weights_matrix[j_idx, marker_idx, :] = weight  # Assign 3D weight

        joint_centers = np.einsum('fmd,jmd->fjd', marker_data_array, weights_matrix)

        return joint_centers
    
    def as_dataframe(self) -> pd.DataFrame:
        df = self.marker_data.time_and_frame_columns.copy()

        for joint_idx, joint_name in enumerate(self.joint_names):
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                col_name = f"{joint_name} {axis}"
                df[col_name] = self.joint_centers[:, joint_idx, axis_idx]

        return df
    
    def as_dataframe_with_unix_timestamps(self, lag_seconds: float = 0) -> pd.DataFrame:
        df = self.as_dataframe()
        df['unix_timestamps'] = df['Time'] + self.marker_data.unix_start_time + lag_seconds
        return df


class DataResampler:
    def __init__(self, data_with_unix_timestamps:pd.DataFrame, freemocap_timestamps: pd.Series):
        self.joint_centers_with_unix_timestamps = data_with_unix_timestamps
        self.freemocap_timestamps = freemocap_timestamps
    
    def resample(self):
        self.resampled_qualisys_data = self._resample(self.joint_centers_with_unix_timestamps, self.freemocap_timestamps)

    def _resample(self, qualisys_df, freemocap_timestamps):
        """
        Resample Qualisys data to match FreeMoCap timestamps using bin averaging.
        
        Parameters:
        -----------
        data_with_unix_timestamps : pandas.DataFrame
            DataFrame with Frame, Time, unix_timestamps and data columns
        freemocap_timestamps : pandas.Series
            Target timestamps to resample to
            
        Returns:
        --------
        pandas.DataFrame
            Resampled data matching freemocap timestamps
        """

        if isinstance(freemocap_timestamps, pd.Series):
            freemocap_timestamps = freemocap_timestamps.to_numpy()
        
        bins = np.append(freemocap_timestamps, freemocap_timestamps[-1] + 
                    (freemocap_timestamps[-1] - freemocap_timestamps[-2]))
    
        # Assign each row to a bin (-1 means it's after the last timestamp)
        qualisys_df['bin'] = pd.cut(qualisys_df['unix_timestamps'], 
                                bins=bins, 
                                labels=range(len(freemocap_timestamps)),
                                include_lowest=True)
        
        # Group by bin and calculate mean
        # Note: dropna=False keeps bins that might be empty
        resampled = qualisys_df.groupby('bin', observed=True).mean(numeric_only=True)
        
        # Handle the last timestamp like the original
        if resampled.index[-1] == len(freemocap_timestamps) - 1:
            last_timestamp = freemocap_timestamps[-1]
            last_frame_data = qualisys_df[qualisys_df['unix_timestamps'] >= last_timestamp].iloc[0]
            resampled.iloc[-1] = last_frame_data[resampled.columns]
        
        resampled_qualisys_data = resampled.reset_index(drop=True)
        
        return resampled_qualisys_data
    
    def _create_marker_array(self) -> np.ndarray:
        """Convert marker data to a NumPy array of shape (frames, markers, 3)."""
        if not hasattr(self, 'resampled_qualisys_data'):
            raise AttributeError("No data available to resample. Resample Qualisys data first.")
        marker_data = self._extract_marker_data(self.resampled_qualisys_data)
        num_frames = len(marker_data)
        num_markers = int(len(marker_data.columns) / 3)

        return marker_data.to_numpy().reshape(num_frames, num_markers, 3)
    
    def _extract_marker_data(self, marker_and_timestamp_dataframe) -> pd.DataFrame:
        """Extract only marker data columns."""
        columns_of_interest = marker_and_timestamp_dataframe.columns[
            ~marker_and_timestamp_dataframe.columns.str.contains(r'^(?:Frame|Time|unix_timestamps|Unnamed)', regex=True)
        ]
        return marker_and_timestamp_dataframe[columns_of_interest]
    
    @property
    def resampled_marker_array(self):
        return self._create_marker_array()
    
    def rotated_resampled_marker_array(self, joint_center_names:List[str]):
        return run_skellyforge_rotation(self.resampled_marker_array, joint_center_names)
    





# class MotionDataRepository:
    
#     def __init__(recording_config: Recording):
#         self.recording_config = recording_config
#         self._validate_required_metadata('qualisys_exported_markers', ['joint_center_weights', 'joint_center_names'])
#         self._initialize_components()
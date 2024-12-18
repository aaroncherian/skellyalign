from skellyalign.temporal_alignment.configs.recording_config import Recording
from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from skellyalign.temporal_alignment.run_skellyforge_rotation import run_skellyforge_rotation
from datetime import datetime

class QualisysTSVProcessor:

    def __init__(self, file_path:Path):
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

    
    def get_unix_start_time(self) -> float:
        """Extract and convert the Qualisys starting timestamp to Unix time."""
        with open(self.file_path, 'r') as file:
            for line in file:
                if line.startswith('TIME_STAMP'):
                    timestamp_str = line.strip().split('\t')[1]
                    datetime_obj = datetime.strptime(timestamp_str, '%Y-%m-%d, %H:%M:%S.%f')
                    return datetime_obj.timestamp()
        raise ValueError(f"No TIME_STAMP found in file: {self.file_path}")

class MarkerDataFrameProcessor:
    """Handles processing of marker data into organized arrays."""

    def __init__(self, marker_data: pd.DataFrame):
        self.marker_data = marker_data

    def get_marker_names(self) -> List[str]:
        """Extract unique marker names from column headers."""
        marker_columns = self._get_marker_columns()
        return list(dict.fromkeys(col.split()[0] for col in marker_columns))

    def create_marker_array(self) -> np.ndarray:
        """Convert marker data to a NumPy array of shape (frames, markers, 3)."""
        marker_data = self.marker_data[self._get_marker_columns()]
        num_frames = len(marker_data)
        num_markers = int(len(marker_data.columns) / 3)
        return marker_data.to_numpy().reshape(num_frames, num_markers, 3)

    def _get_marker_columns(self) -> pd.Index:
        """Get columns containing marker data."""
        return self.marker_data.columns[
            ~self.marker_data.columns.str.contains(r'^(?:Frame|Time|unix_timestamps|Unnamed)', regex=True)
        ]


class JointCenterCalculator:
    def __init__(self, 
                 marker_array: np.ndarray, 
                 marker_names: List[str],
                 weights: Dict[str, Dict[str, List[float]]]):
        self.marker_array = marker_array
        self.marker_names = marker_names
        self.weights = weights
        
    def calculate(self) -> Tuple[np.ndarray, List[str]]:
        num_frames, num_markers, _ = self.marker_array.shape
        num_joint_centers = len(self.weights)

        marker_to_index = {marker: i for i, marker in enumerate(self.marker_names)}
        weights_matrix = np.zeros((num_joint_centers, num_markers, 3))

        for j_idx, (joint, markers_weights) in enumerate(self.weights.items()):
            for marker, weight in markers_weights.items():
                marker_idx = marker_to_index[marker]
                weights_matrix[j_idx, marker_idx, :] = weight


        joint_centers = np.einsum('fmd,jmd->fjd', self.marker_array, weights_matrix)
        
        return joint_centers
    
class QualisysData:
    def __init__(self, recording_config: Recording):
        self.config = recording_config
        self._validate_required_metadata('qualisys_exported_markers', ['joint_center_names', 'joint_center_weights'])

        self._tsv_processor = QualisysTSVProcessor(file_path=self.config.get_component_file_path('qualisys_exported_markers', 'markers'))

    def _validate_required_metadata(self, component_name: str, required_keys: List[str]):
        component = self.config.components[component_name]
        for key in required_keys:
            if not component.metadata.has(key):
                raise ValueError(f"Missing required metadata '{key}' for component {component_name}")
    
    def get_marker_data(self) -> pd.DataFrame:
        return self._tsv_processor.load_tsv()
    
    def get_marker_processor(self) -> MarkerDataFrameProcessor:
        return MarkerDataFrameProcessor(marker_data=self.get_marker_data())

    def calculate_joint_centers(self) -> np.ndarray:
        marker_processor = self.get_marker_processor()

        calculator = JointCenterCalculator(
            marker_array=marker_processor.create_marker_array(),
            marker_names=marker_processor.get_marker_names(),
            weights=self.config.components['qualisys_exported_markers'].metadata.get('joint_center_weights')
        )

        return calculator.calculate()

    def get_unix_start_time(self) -> float:
        return self._tsv_processor.get_unix_start_time()
    
    def create_dataframe_with_unix_timestamps(self, lag_in_seconds: float = 0) -> pd.DataFrame:
        marker_dataframe = self.get_marker_data()
        joint_names = self.config.components['qualisys_exported_markers'].metadata.get('joint_center_names')
        joint_centers = self.calculate_joint_centers()

        joint_df = pd.DataFrame({
            'Frame': marker_dataframe['Frame'],
            'Time': marker_dataframe['Time'],
        })

        for joint_idx, joint_name in enumerate(joint_names):
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                col_name = f"{joint_name} {axis}"
                joint_df[col_name] = joint_centers[:, joint_idx, axis_idx]

        lag_adjusted_dataframe = joint_df.copy()

class QualisysDataProcessor:
    def __init__(self, marker_and_timestamp_df: pd.DataFrame, joint_center_weights: Dict[str, Dict[str, List[float]]]):
        self.marker_and_timestamp_df = marker_and_timestamp_df
        self.joint_center_weights = joint_center_weights
        self.marker_names = self._extract_marker_names()
        self.marker_array = self._create_marker_array()
        self.joint_centers = None

    def _extract_marker_data(self) -> pd.DataFrame:
        """Extract only marker data columns."""
        columns_of_interest = self.marker_and_timestamp_df.columns[
            ~self.marker_and_timestamp_df.columns.str.contains(r'^(?:Frame|Time|unix_timestamps|Unnamed)', regex=True)
        ]
        return self.marker_and_timestamp_df[columns_of_interest]

    def _extract_marker_names(self) -> List[str]:
        """Extract unique marker names from column headers."""
        marker_columns = self._extract_marker_data().columns
        return list(dict.fromkeys(col.split()[0] for col in marker_columns))

    def _create_marker_array(self) -> np.ndarray:
        """Convert marker data to a NumPy array of shape (frames, markers, 3)."""
        marker_data = self._extract_marker_data()
        num_frames = len(marker_data)
        num_markers = int(len(marker_data.columns) / 3)
        return marker_data.to_numpy().reshape(num_frames, num_markers, 3)

    def calculate_joint_centers(self):
        """
        Optimized calculation of joint centers for Qualisys data with 3D weights.

        Parameters:
            marker_array (np.ndarray): Shape (num_frames, num_markers, 3), 3D marker data.
            joint_center_weights (dict): Weights for each joint as {joint_name: {marker_name: [weight_x, weight_y, weight_z]}}.
            marker_names (list): List of marker names corresponding to marker_array.

        Result:
            np.ndarray: Joint centers with shape (num_frames, num_joints, 3).
        """
        num_frames, num_markers, _ = self.marker_array.shape
        num_joints = len(self.joint_center_weights)

        marker_to_index = {marker: i for i, marker in enumerate(self.marker_names)}

        weights_matrix = np.zeros((num_joints, num_markers, 3))
        for j_idx, (joint, markers_weights) in enumerate(self.joint_center_weights.items()):
            for marker, weight in markers_weights.items():
                marker_idx = marker_to_index[marker]
                weights_matrix[j_idx, marker_idx, :] = weight  # Assign 3D weight

        self.joint_centers = np.einsum('fmd,jmd->fjd', self.marker_array, weights_matrix)

        return self.joint_centers


    def create_joint_center_df(self) -> pd.DataFrame:
        """
        Create DataFrame from joint center data with timestamps .
        
        Parameters:
            joint_center_array: Array of joint center positions
            joint_names: Names of the joints
            timing_df: DataFrame containing Frame and Time columns
        """
        if self.joint_centers is None:
            raise ValueError("Joint centers have not been calculated. Call calculate_joint_centers() first.")

        joint_names = list(self.joint_center_weights.keys())
        joint_df = pd.DataFrame({
            'Frame': self.marker_and_timestamp_df['Frame'],
            'Time': self.marker_and_timestamp_df['Time'],
        })

        for joint_idx, joint_name in enumerate(joint_names):
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                col_name = f"{joint_name} {axis}"
                joint_df[col_name] = self.joint_centers[:, joint_idx, axis_idx]

        return joint_df

    def add_unix_timestamps(self, df:pd.DataFrame, unix_start_timestamp: float, lag_in_seconds: float = 0) -> pd.DataFrame:
        """
        Insert a new column with Unix timestamps to a copy of the Qualisys dataframe.
        
        Parameters:
            df (pd.DataFrame): The original Qualisys dataframe with a 'Time' column in seconds.
            start_timestamp (str): The Qualisys start time as a string in the format '%Y-%m-%d, %H:%M:%S.%f'.
            lag_in_seconds (float, optional): The lag between Qualisys and FreeMoCap data in seconds. Default is 0.
            
        Returns:
            pd.DataFrame: A copy of the original Qualisys dataframe with a new 'unix_timestamps' column.
        """
        df = df.copy()
        adjusted_time = df['Time'] + lag_in_seconds
        df['unix_timestamps'] = adjusted_time + unix_start_timestamp
        return df
    
    def create_dataframe_with_unix_timestamps(self, unix_start_time: float, lag_in_seconds: float = 0) -> pd.DataFrame:
        joint_center_df = self.create_joint_center_df()
        return self.add_unix_timestamps(joint_center_df, unix_start_time, lag_in_seconds)


class FreeMoCapArrayAdaptor:
    def __init__(self, recording_config: Recording, model_to_use = 'mediapipe'):
        self.recording_config = recording_config
        self._validate_required_metadata(model_to_use, ['body'])
        self.file_path = self.recording_config.get_component_file_path(
            model_to_use, 'body')
        
    def _validate_required_metadata(self, component_name: str, required_keys: List[str]):
        component = self.recording_config.components[component_name]
        for key in required_keys:
            if not component.metadata.has(key):
                raise ValueError(f"Missing required metadata '{key}' for component {component_name}")
            
    def load_array(self) -> np.ndarray:
        return np.load(self.file_path)
    
class FreeMoCapArrayProcessor:
    def __init__(self, joint_center_data: np.ndarray, joint_center_names: List[str]):
        self.joint_center_data = joint_center_data
        self.joint_center_names = joint_center_names

    def process(self) -> np.ndarray:
        return run_skellyforge_rotation(
            raw_skeleton_data=self.joint_center_data, 
            landmark_names=self.joint_center_names)
    



# class MotionDataRepository:
    
#     def __init__(recording_config: Recording):
#         self.recording_config = recording_config
#         self._validate_required_metadata('qualisys_exported_markers', ['joint_center_weights', 'joint_center_names'])
#         self._initialize_components()
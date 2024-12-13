from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict
import numpy as np

def get_header_length(file_path: Path) -> int:
    """Determine the length of the header in a Qualisys TSV file."""
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if line.startswith('TRAJECTORY_TYPES'):
                return i + 1
    return 0

def get_unix_start_time(file_path: Path) -> float:
    """Extract and convert the Qualisys starting timestamp to Unix time."""
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('TIME_STAMP'):
                timestamp_str = line.strip().split('\t')[1]
                datetime_obj = datetime.strptime(timestamp_str, '%Y-%m-%d, %H:%M:%S.%f')
                return datetime_obj.timestamp()
    raise ValueError(f"No TIME_STAMP found in file: {file_path}")

def load_qualisys_data(file_path: Path) -> Tuple[pd.DataFrame, float]:
    """Load Qualisys marker data and extract start time."""
    header_length = get_header_length(file_path)
    
    marker_trajectories = pd.read_csv(
        file_path, 
        delimiter='\t', 
        skiprows=header_length
    )
    
    return marker_trajectories

def extract_just_marker_data_from_dataframe(marker_df: pd.DataFrame) -> pd.Index:
    """Extract columns containing marker data, excluding metadata columns."""
    columns_of_interest = marker_df.columns[
        ~marker_df.columns.str.contains(r'^(?:Frame|Time|unix_timestamps|Unnamed)', regex=True)
    ]
    
    return marker_df[columns_of_interest]

def extract_marker_names(marker_dataframe:pd.DataFrame) -> List[str]:
    """Extract unique marker names from column headers."""
    return list(dict.fromkeys(col.split()[0] for col in marker_dataframe.columns))

def create_marker_array(marker_df: pd.DataFrame) -> np.ndarray:
    """Convert marker data to numpy array of shape (frames, markers, 3)."""
    marker_data_flat = marker_df.to_numpy()
    num_frames = len(marker_df)
    num_markers = int(len(marker_df.columns)/3)
    return marker_data_flat.reshape(num_frames, num_markers, 3)

def calculate_joint_centers(marker_array, joint_center_weights, marker_names):
    """
    Optimized calculation of joint centers for Qualisys data with 3D weights.

    Parameters:
        marker_array (np.ndarray): Shape (num_frames, num_markers, 3), 3D marker data.
        joint_center_weights (dict): Weights for each joint as {joint_name: {marker_name: [weight_x, weight_y, weight_z]}}.
        marker_names (list): List of marker names corresponding to marker_array.

    Returns:
        np.ndarray: Joint centers with shape (num_frames, num_joints, 3).
    """
    num_frames, num_markers, _ = marker_array.shape
    num_joints = len(joint_center_weights)

    # Create a mapping from marker names to indices
    marker_to_index = {marker: i for i, marker in enumerate(marker_names)}

    # Initialize weight matrix (num_joints, num_markers, 3)
    weights_matrix = np.zeros((num_joints, num_markers, 3))
    for j_idx, (joint, markers_weights) in enumerate(joint_center_weights.items()):
        for marker, weight in markers_weights.items():
            marker_idx = marker_to_index[marker]
            weights_matrix[j_idx, marker_idx, :] = weight  # Assign 3D weight

    # Compute joint centers
    # (num_frames, num_joints, 3) = (num_frames, num_markers, 3) @ (num_joints, num_markers, 3).T
    joint_centers = np.einsum('fmd,jmd->fjd', marker_array, weights_matrix)

    return joint_centers


def create_joint_center_df(joint_center_array: np.ndarray,
                         joint_names: List[str],
                         dataframe_with_timestamps: pd.DataFrame) -> pd.DataFrame:
    """
    Create DataFrame from joint center data with timestamps .
    
    Args:
        joint_center_array: Array of joint center positions
        joint_names: Names of the joints
        timing_df: DataFrame containing Frame and Time columns
    """
    joint_df = pd.DataFrame({
        'Frame': dataframe_with_timestamps['Frame'],
        'Time': dataframe_with_timestamps['Time'],
    })
    
    for joint_idx, joint_name in enumerate(joint_names):
        for axis_idx, axis in enumerate(['x', 'y', 'z']):
            col_name = f"{joint_name} {axis}"
            joint_df[col_name] = joint_center_array[:, joint_idx, axis_idx]
            
    return joint_df

def create_and_insert_unix_timestamp_column(df, unix_start_timestamp, lag_in_seconds=0):
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
    # Adjust the 'Time' column based on the calculated lag in seconds
    adjusted_time = df['Time'] + lag_in_seconds
    
    # Insert the new column with Unix timestamps
    df.insert(df.columns.get_loc('Time') + 1, 'unix_timestamps', adjusted_time + unix_start_timestamp)
    
    return df
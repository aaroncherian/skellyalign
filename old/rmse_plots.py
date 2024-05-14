import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def calculate_rmse_per_frame(freemocap_df, qualisys_df):
    """
    Calculate RMSE per frame for each marker and each dimension.
    
    Parameters:
    - freemocap_df (pandas.DataFrame): DataFrame containing FreeMoCap data per frame for each marker and each dimension (x, y, z).
    - qualisys_df (pandas.DataFrame): DataFrame containing Qualisys data per frame for each marker and each dimension (x, y, z).
    
    Returns:
    - rmse_per_frame_df (pandas.DataFrame): DataFrame containing RMSE per frame for each marker and each dimension (x, y, z).
    """
    
    # Initialize an empty list to hold the data
    rmse_data = []

    # Loop through each unique marker name in the FreeMoCap DataFrame
    for marker_name in freemocap_df['Marker'].unique():
        
        if marker_name in qualisys_df['Marker'].unique():
            # Filter the data for the specific marker in both DataFrames
            freemocap_marker_df = freemocap_df[freemocap_df['Marker'] == marker_name]
            qualisys_marker_df = qualisys_df[qualisys_df['Marker'] == marker_name]
            
            # Merge the two DataFrames on 'Frame' for alignment
            merged_df = pd.merge(freemocap_marker_df, qualisys_marker_df, on='Frame', suffixes=('_freemocap', '_qualisys'))
            
            # Calculate RMSE for each dimension (x, y, z)
            for dim in ['x', 'y', 'z']:
                rmse_per_frame = np.sqrt((merged_df[f"{dim}_freemocap"] - merged_df[f"{dim}_qualisys"]) ** 2)
                
                # Append the results to the list
                for frame, rmse_value in zip(merged_df['Frame'], rmse_per_frame):
                    frame_data = [frame, marker_name]
                    frame_data.extend([rmse_value if dim == d else None for d in ['x', 'y', 'z']])
                    rmse_data.append(frame_data)
                    
    # Create a DataFrame from the list
    columns = ['Frame', 'Marker', 'x', 'y', 'z']
    rmse_per_frame_df = pd.DataFrame(rmse_data, columns=columns)
    
    # Remove any None values to clean up the DataFrame
    rmse_per_frame_df = rmse_per_frame_df.groupby(['Frame', 'Marker']).first().reset_index()
    
    return rmse_per_frame_df



def calculate_max_min_errors(qualisys_data, freemocap_data, qualisys_indices, mediapipe_indices):
    """
    Calculate the maximum and minimum errors per marker.
    
    Parameters:
        qualisys_data (np.ndarray): The reference 3D data from Qualisys.
        freemocap_data (np.ndarray): The 3D data from FreeMoCap to be compared.
        marker_indices (list): List of marker names.
    
    Returns:
        max_min_errors_df (pd.DataFrame): A DataFrame containing maximum and minimum errors per marker.
    """
    max_min_errors_list = []
    dimensions = ['x', 'y', 'z']
    
    for marker_name in mediapipe_indices:
        if marker_name in qualisys_indices:  # Only run if the marker is in both Qualisys and FreeMoCap
            mediapipe_marker_index = mediapipe_indices.index(marker_name)
            qualisys_marker_index = qualisys_indices.index(marker_name)
            
            errors = freemocap_data[:, mediapipe_marker_index, :] - qualisys_data[:, qualisys_marker_index, :]
            
            max_errors = np.max(errors, axis=0)
            min_errors = np.min(errors, axis=0)
            
            for dim, max_err, min_err in zip(dimensions, max_errors, min_errors):
                max_min_errors_list.append({'marker': marker_name, 'dimension': dim, 'max_error': max_err, 'min_error': min_err})
    
    max_min_errors_df = pd.DataFrame(max_min_errors_list)
    return max_min_errors_df

def plot_trajectory_with_error_shading(freemocap_df, qualisys_df, rmse_per_frame_df, joint_name, dimensions=['x']):
    """
    Plot the trajectory for a specific joint across X, Y, Z dimensions with error shading.
    
    Parameters:
    - freemocap_df (pandas.DataFrame): DataFrame containing FreeMoCap data per frame for each marker and each dimension (x, y, z).
    - rmse_per_frame_df (pandas.DataFrame): DataFrame containing RMSE per frame for each marker and each dimension (x, y, z).
    - joint_name (str): The name of the joint to plot.
    - dimensions (list): List of dimensions to plot, default is ['x', 'y', 'z'].

    """
    # Filter the DataFrames to only include data for the specified joint
    freemocap_joint_trajectory_data = freemocap_df[freemocap_df['Marker'] == joint_name]
    qualisys_joint_trajectory_data = qualisys_df[qualisys_df['Marker'] == joint_name]
    joint_rmse_data = rmse_per_frame_df[rmse_per_frame_df['Marker'] == joint_name]
    
    fig, axes = plt.subplots(len(dimensions), 1, figsize=(10, 6))
    
    for ax, dim in zip(axes, dimensions):
        # Further filter to get data for the specific dimension
        dim_trajectory_data = freemocap_joint_trajectory_data[dim]
        dim_rmse_data = joint_rmse_data[dim]
        
        # Calculate percentiles for shading
        p25 = dim_rmse_data.quantile(0.25)
        p50 = dim_rmse_data.quantile(0.50)
        p75 = dim_rmse_data.quantile(0.75)
        
        ax.plot(freemocap_joint_trajectory_data['Frame'], dim_trajectory_data, label=f'FreeMoCap Trajectory ({dim.upper()})')
        ax.plot(qualisys_joint_trajectory_data['Frame'], qualisys_joint_trajectory_data[dim], label=f'Qualisys Trajectory({dim.upper()})')
        ax.fill_between(freemocap_joint_trajectory_data['Frame'], dim_trajectory_data.min(), dim_trajectory_data, where=(dim_rmse_data >= p75), alpha=0.5, color='red')
        ax.fill_between(freemocap_joint_trajectory_data['Frame'], dim_trajectory_data.min(), dim_trajectory_data, where=(dim_rmse_data <= p25), alpha=0.5, color='green')
        # ax.fill_between(freemocap_joint_trajectory_data['Frame'], dim_trajectory_data.min(), dim_trajectory_data, where=(dim_rmse_data > p25) & (dim_rmse_data < p75), alpha=0.5, color='yellow')
        
        ax.set_title(f"{joint_name} - {dim.upper()} Dimension")
        ax.set_xlabel('Frame')
        ax.set_ylabel('Trajectory')
        ax.legend()
        
    plt.tight_layout()
    plt.show()
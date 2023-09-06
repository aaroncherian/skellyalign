import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_rmse_per_frame(freemocap_data, qualisys_data, mediapipe_indices, qualisys_indices):
    """
    Calculate RMSE per frame for each marker and each dimension.
    
    Parameters:
    - freemocap_data (numpy.ndarray): The FreeMoCap data array. Shape should be (num_frames, num_markers, 3).
    - qualisys_data (numpy.ndarray): The Qualisys data array. Shape should be (num_frames, num_markers, 3).
    - mediapipe_indices (list): List of marker names in FreeMoCap data.
    - qualisys_indices (list): List of marker names in Qualisys data.
    
    Returns:
    - rmse_per_frame_df (pandas.DataFrame): DataFrame containing RMSE per frame for each marker and each dimension (x, y, z).
    """
    
    num_frames = freemocap_data.shape[0]
    dimensions = ['x', 'y', 'z']
    
    # Initialize an empty list to hold the data
    rmse_data = []
    
    # Loop through each marker name
    for marker_name in mediapipe_indices:
        if marker_name in qualisys_indices:
            mediapipe_marker_index = mediapipe_indices.index(marker_name)
            qualisys_marker_index = qualisys_indices.index(marker_name)
            
            # Loop through each dimension (x, y, z)
            for dim_index, dim_name in enumerate(dimensions):
                freemocap_series = freemocap_data[:, mediapipe_marker_index, dim_index]
                qualisys_series = qualisys_data[:, qualisys_marker_index, dim_index]
                
                # Calculate RMSE for each frame
                rmse_per_frame = np.sqrt((freemocap_series - qualisys_series)**2)
                
                # Append the results to the list
                for frame_index, rmse_value in enumerate(rmse_per_frame):
                    rmse_data.append([frame_index, marker_name, dim_name, rmse_value])
    
    # Create a DataFrame from the list
    rmse_per_frame_df = pd.DataFrame(rmse_data, columns=['Frame', 'Marker', 'Dimension', 'RMSE'])
    
    return rmse_per_frame_df

def plot_error_heatmap(rmse_per_frame_df):
    """
    Plot a heatmap of RMSE values per frame for each marker and each dimension.
    
    Parameters:
    - rmse_per_frame_df (pandas.DataFrame): DataFrame containing RMSE per frame for each marker and each dimension.
    
    Returns:
    - None (displays the heatmap plot).
    """
    
    # Create a pivot table for the heatmap
    pivot_df = rmse_per_frame_df.pivot_table(values='RMSE', index=['Frame'], columns=['Marker', 'Dimension'], aggfunc=np.mean)
    
    # Plot the heatmap
    plt.figure(figsize=(15, 15))
    sns.heatmap(pivot_df, cmap='coolwarm', cbar_kws={'label': 'RMSE'})
    plt.title('Heatmap of RMSE per Frame for Each Marker and Dimension')
    plt.show()

# Generate some example data for demonstration
# Let's assume we have 100 frames, 3 markers, and 3 dimensions (x, y, z)
# num_frames = 100
# num_markers = 3
# num_dims = 3

# # For simplicity, we'll generate random data for FreeMoCap and Qualisys
# freemocap_data = np.random.rand(num_frames, num_markers, num_dims)
# qualisys_data = np.random.rand(num_frames, num_markers, num_dims)

# # Assume these are the names of the markers in FreeMoCap and Qualisys
# mediapipe_indices = ['marker_1', 'marker_2', 'marker_3']
# qualisys_indices = ['marker_1', 'marker_2', 'marker_3']

# # Calculate RMSE per frame
# rmse_per_frame_df = calculate_rmse_per_frame(freemocap_data, qualisys_data, mediapipe_indices, qualisys_indices)

# # Plot the heatmap
# plot_error_heatmap(rmse_per_frame_df)

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
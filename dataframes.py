import numpy as np
import pandas as pd

def calculate_rmse_per_frame(freemocap_dataframe, qualisys_dataframe, mediapipe_indices, qualisys_indices):
    """
    Calculate RMSE per frame for each marker and each dimension.
    
    Parameters:
    - freemocap_data (numpy.ndarray): The FreeMoCap data array. Shape should be (num_frames, num_markers, 3).
    - qualisys_data (numpy.ndarray): The Qualisys data array. Shape should be (num_frames, num_markers, 3).
    - mediapipe_indices (list): List of marker names in FreeMoCap data.
    - qualisys_indices (list): List of marker names in Qualisys data.
    
    Returns:
    - rmse_per_frame_df (pandas.DataFrame): DataFrame containing RMSE per frame for each marker and each dimension (X, Y, Z).
    """
    
    # num_frames = freemocap_data.shape[0]
    dimensions = ['x', 'y', 'z']
    
    # Initialize an empty list to hold the data
    rmse_data = []
    
    # Loop through each marker name
    for marker_name in freemocap_dataframe['Marker'].unique():
        if marker_name in qualisys_dataframe['Marker'].unique():
            freemocap_marker_data = freemocap_dataframe[freemocap_dataframe['Marker'] == marker_name]
            qualisys_marker_data = qualisys_dataframe[qualisys_dataframe['Marker'] == marker_name]
            
            # Loop through each dimension (X, Y, Z)
            for dim_index, dim_name in enumerate(dimensions):

        
                # Calculate RMSE for each frame
                rmse_per_frame = np.sqrt((freemocap_series - qualisys_series)**2)
                
                # Append the results to the list
                for frame_index, rmse_value in enumerate(rmse_per_frame):
                    frame_data = [frame_index, marker_name]
                    frame_data.extend([rmse_value if dim_name == d else None for d in dimensions])
                    rmse_data.append(frame_data)
    
    # Create a DataFrame from the list
    columns = ['Frame', 'Marker'] + dimensions
    rmse_per_frame_df = pd.DataFrame(rmse_data, columns=columns)

    # Remove any None values to clean up the DataFrame
    rmse_per_frame_df = rmse_per_frame_df.groupby(['Frame', 'Marker']).first().reset_index()

    return rmse_per_frame_df
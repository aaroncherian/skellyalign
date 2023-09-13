from data_utils.load_and_process_data import load_and_process_data

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def calculate_squared_error(value1, value2):
    return (value1 - value2) ** 2

def calculate_rmse_from_squared_errors(squared_errors):
    mean_squared_error = np.mean(squared_errors)
    root_mean_squared_error = np.sqrt(mean_squared_error)
    return root_mean_squared_error

def calculate_squared_error_single_df(dataframe_of_3d_data):
    """
    Calculate the squared error between the 'freemocap' and 'qualisys' systems
    for each frame and marker.

    Parameters:
    - dataframe_of_3d_data (pd.DataFrame): A DataFrame containing 3D motion capture data from both systems.
      The DataFrame should contain columns ['frame', 'marker', 'x', 'y', 'z', 'system'].

    Returns:
    - pd.DataFrame: A new DataFrame containing only the squared error calculations.
    """
    
    # Pivot the DataFrame so that the 'system' becomes new columns
    pivot_df = dataframe_of_3d_data.pivot_table(index=['frame', 'marker'], columns='system', values=['x', 'y', 'z'])
    
    
    # Calculate squared errors for x, y, z dimensions
    error_df = pd.DataFrame()
    error_df['frame'] = pivot_df.index.get_level_values('frame')
    error_df['marker'] = pivot_df.index.get_level_values('marker')

    
    # Loop to calculate squared errors
    for coord in ['x', 'y', 'z']:
        error_df[f'{coord}_error'] = calculate_squared_error(pivot_df[coord]['freemocap'], pivot_df[coord]['qualisys']).reset_index(drop=True) #need to reset index here to make sure there's not a dataframe index mismatch
        
    return error_df

def calculate_absolute_error_from_squared(squared_error_df):
    """
    Calculate the absolute error based on the squared errors for each marker and frame.

    Parameters:
    - squared_error_df (pd.DataFrame): A DataFrame containing squared error calculations.
      The DataFrame should contain columns ['frame', 'marker', 'x_error', 'y_error', 'z_error'].

    Returns:
    - pd.DataFrame: A new DataFrame containing only the absolute error calculations.
    """
    
    # Calculate absolute error by taking the square root of squared error for each coordinate
    absolute_error_df = pd.DataFrame()
    absolute_error_df['frame'] = squared_error_df['frame']
    absolute_error_df['marker'] = squared_error_df['marker']
    
    # Loop to calculate absolute errors
    for coord in ['x', 'y', 'z']:
        absolute_error_df[f'{coord}_error'] = np.sqrt(squared_error_df[f'{coord}_error'])
        
    return absolute_error_df

def calculate_rmse_from_dataframe(squared_error_df):
    # Create an empty DataFrame to store the RMSE values
    rmse_df = pd.DataFrame()

    # Calculate RMSE for each joint
    #groupby partitions the dataframe into smaller groups based on the marker column
    rmse_joints = squared_error_df.groupby('marker')[['x_error', 'y_error', 'z_error']].apply(calculate_rmse_from_squared_errors).reset_index()
    rmse_joints['dimension'] = 'Per Joint'
    rmse_joints = rmse_joints.melt(id_vars=['marker', 'dimension'], var_name='coordinate', value_name='RMSE')
    
    # Append to rmse_df
    rmse_df = pd.concat([rmse_df, rmse_joints], ignore_index=True)

    # Calculate RMSE for each dimension
    rmse_dimensions = squared_error_df[['x_error', 'y_error', 'z_error']].apply(calculate_rmse_from_squared_errors).reset_index()
    rmse_dimensions.columns = ['coordinate', 'RMSE']
    rmse_dimensions['dimension'] = 'Per Dimension'
    rmse_dimensions['marker'] = 'All'
    
    # Append to rmse_df
    rmse_df = pd.concat([rmse_df, rmse_dimensions], ignore_index=True)

    # Calculate overall RMSE (turn the datafrom into a 1d array with all the error values)
    overall_rmse = calculate_rmse_from_squared_errors(squared_error_df[['x_error', 'y_error', 'z_error']].values.flatten())
    overall_rmse = pd.DataFrame({'marker': 'All', 'dimension': 'Overall', 'coordinate': 'All', 'RMSE': [overall_rmse]})
    
    # Append to rmse_df
    rmse_df = pd.concat([rmse_df, overall_rmse], ignore_index=True)

    return rmse_df


qualisys_data_path = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3\output_data\clipped_qualisys_skel_3d.npy")
freemocap_data_path = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\mediapipe_body_3d_xyz_transformed.npy")

dataframe_of_3d_data = load_and_process_data(path_to_freemocap_data=freemocap_data_path, path_to_qualisys_data=qualisys_data_path)

squared_error_df = calculate_squared_error_single_df(dataframe_of_3d_data=dataframe_of_3d_data)
absolute_error_df = calculate_absolute_error_from_squared(squared_error_df=squared_error_df)
rmse_result = calculate_rmse_from_dataframe(squared_error_df)

f =2 


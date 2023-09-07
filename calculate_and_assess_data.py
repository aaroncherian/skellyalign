from main import main
from RMSE_GUI import run_GUI
import numpy as np
from freemocap_utils.GUI_widgets.rmse_widgets.RMSE_calculator import calculate_rmse_dataframe, calculate_rmse_per_timepoint_per_dimension
from rmse_plots import calculate_rmse_per_frame, plot_error_heatmap, calculate_max_min_errors
from marker_lists.mediapipe_markers import mediapipe_markers
from marker_lists.qualisys_markers import qualisys_markers

import pandas as pd

qualisys_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3\output_data\clipped_qualisys_skel_3d.npy"
freemocap_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\mediapipe_body_3d_xyz.npy"

freemocap_data = np.load(freemocap_data_path)
qualisys_data = np.load(qualisys_data_path)


freemocap_data_transformed = main(freemocap_data=freemocap_data, qualisys_data=qualisys_data, representative_frame=300)

rmse_dataframe = calculate_rmse_dataframe(qualisys_data=qualisys_data, freemocap_data=freemocap_data_transformed)

rmse_per_frame = calculate_rmse_per_frame(freemocap_data=freemocap_data_transformed, qualisys_data=qualisys_data, mediapipe_indices=mediapipe_markers, qualisys_indices=qualisys_markers)
# plot_error_heatmap(rmse_per_frame)

min_max_errors = calculate_max_min_errors(qualisys_data=qualisys_data, freemocap_data=freemocap_data_transformed, qualisys_indices=qualisys_markers, mediapipe_indices=mediapipe_markers)

def convert_to_dataframe(freemocap_data, mediapipe_indices):
    """
    Convert the FreeMoCap data from a numpy array to a pandas DataFrame.

    Parameters:
    - freemocap_data (numpy.ndarray): The FreeMoCap data array. Shape should be (num_frames, num_markers, 3).
    - mediapipe_indices (list): List of marker names in FreeMoCap data.

    Returns:
    - freemocap_df (pandas.DataFrame): DataFrame containing FreeMoCap data with columns ['Frame', 'Marker', 'X', 'Y', 'Z'].

    """
    num_frames = freemocap_data.shape[0]
    num_markers = freemocap_data.shape[1]

    frame_list = []
    marker_list = []
    x_list = []
    y_list = []
    z_list = []

    for frame in range(num_frames):
        for marker in range(num_markers):
            frame_list.append(frame)
            marker_list.append(mediapipe_indices[marker])
            x_list.append(freemocap_data[frame, marker, 0])
            y_list.append(freemocap_data[frame, marker, 1])
            z_list.append(freemocap_data[frame, marker, 2])

    freemocap_df = pd.DataFrame({
        'Frame': frame_list,
        'Marker': marker_list,
        'x': x_list,
        'y': y_list,
        'z': z_list
    })

    return freemocap_df

freemocap_df = convert_to_dataframe(freemocap_data=freemocap_data_transformed, mediapipe_indices=mediapipe_markers)

f = 2


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Create a function for plotting
def plot_trajectory_with_error_shading(freemocap_df, rmse_per_frame_df, joint_name, dimensions=['x', 'y', 'z']):
    """
    Plot the trajectory for a specific joint across X, Y, Z dimensions with error shading.
    
    Parameters:
    - freemocap_df (pandas.DataFrame): DataFrame containing FreeMoCap data per frame for each marker and each dimension (x, y, z).
    - rmse_per_frame_df (pandas.DataFrame): DataFrame containing RMSE per frame for each marker and each dimension (x, y, z).
    - joint_name (str): The name of the joint to plot.
    - dimensions (list): List of dimensions to plot, default is ['x', 'y', 'z'].

    """
    # Filter the DataFrames to only include data for the specified joint
    joint_trajectory_data = freemocap_df[freemocap_df['Marker'] == joint_name]
    joint_rmse_data = rmse_per_frame_df[rmse_per_frame_df['Marker'] == joint_name]
    
    fig, axes = plt.subplots(len(dimensions), 1, figsize=(10, 6))
    
    for ax, dim in zip(axes, dimensions):
        # Further filter to get data for the specific dimension
        dim_trajectory_data = joint_trajectory_data[dim]
        dim_rmse_data = joint_rmse_data[dim]
        
        # Calculate percentiles for shading
        p25 = dim_rmse_data.quantile(0.25)
        p50 = dim_rmse_data.quantile(0.50)
        p75 = dim_rmse_data.quantile(0.75)
        
        ax.plot(joint_trajectory_data['Frame'], dim_trajectory_data, label=f'Trajectory ({dim.upper()})')
        ax.fill_between(joint_trajectory_data['Frame'], dim_trajectory_data.min(), dim_trajectory_data, where=(dim_rmse_data >= p75), alpha=0.5, color='red')
        ax.fill_between(joint_trajectory_data['Frame'], dim_trajectory_data.min(), dim_trajectory_data, where=(dim_rmse_data <= p25), alpha=0.5, color='green')
        ax.fill_between(joint_trajectory_data['Frame'], dim_trajectory_data.min(), dim_trajectory_data, where=(dim_rmse_data > p25) & (dim_rmse_data < p75), alpha=0.5, color='yellow')
        
        ax.set_title(f"{joint_name} - {dim.upper()} Dimension")
        ax.set_xlabel('Frame')
        ax.set_ylabel('Trajectory')
        ax.legend()
        
    plt.tight_layout()
    plt.show()
plot_trajectory_with_error_shading(freemocap_df=freemocap_df, rmse_per_frame_df=rmse_per_frame, joint_name='left_heel', dimensions=['x', 'y', 'z'])
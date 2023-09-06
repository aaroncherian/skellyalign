from main import main
from RMSE_GUI import run_GUI
import numpy as np
from freemocap_utils.GUI_widgets.rmse_widgets.RMSE_calculator import calculate_rmse_dataframe, calculate_rmse_per_timepoint_per_dimension
from rmse_plots import calculate_rmse_per_frame, plot_error_heatmap, calculate_max_min_errors
from marker_lists.mediapipe_markers import mediapipe_markers
from marker_lists.qualisys_markers import qualisys_markers

qualisys_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3\output_data\clipped_qualisys_skel_3d.npy"
freemocap_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\mediapipe_body_3d_xyz.npy"

freemocap_data = np.load(freemocap_data_path)
qualisys_data = np.load(qualisys_data_path)


freemocap_data_transformed = main(freemocap_data=freemocap_data, qualisys_data=qualisys_data, representative_frame=800)

rmse_dataframe = calculate_rmse_dataframe(qualisys_data=qualisys_data, freemocap_data=freemocap_data_transformed)

rmse_per_frame = calculate_rmse_per_frame(freemocap_data=freemocap_data_transformed, qualisys_data=qualisys_data, mediapipe_indices=mediapipe_markers, qualisys_indices=qualisys_markers)
# plot_error_heatmap(rmse_per_frame)

min_max_errors = calculate_max_min_errors(qualisys_data=qualisys_data, freemocap_data=freemocap_data_transformed, qualisys_indices=qualisys_markers, mediapipe_indices=mediapipe_markers)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Create a function for plotting
def plot_error_trajectory(rmse_per_frame_df, joint_name, dimensions=['x', 'y', 'z']):
    """
    Plot the error trajectory for a specific joint across X, Y, Z dimensions.
    
    Parameters:
    - rmse_per_frame_df (pandas.DataFrame): DataFrame containing RMSE per frame for each marker and each dimension (x, y, z).
    - joint_name (str): The name of the joint to plot.
    - dimensions (list): List of dimensions to plot, default is ['x', 'y', 'z'].

    """
    # Filter the DataFrame to only include data for the specified joint
    joint_data = rmse_per_frame_df[rmse_per_frame_df['Marker'] == joint_name]
    
    fig, axes = plt.subplots(len(dimensions), 1, figsize=(10, 6))
    
    for ax, dim in zip(axes, dimensions):
        # Further filter to get data for the specific dimension
        dim_data = joint_data[joint_data['Dimension'] == dim]
        
        # Calculate percentiles for shading
        p25 = dim_data['RMSE'].quantile(0.25)
        p50 = dim_data['RMSE'].quantile(0.50)
        p75 = dim_data['RMSE'].quantile(0.75)
        
        ax.plot(dim_data['Frame'], dim_data['RMSE'], label=f'RMSE ({dim.upper()})')
        ax.fill_between(dim_data['Frame'], 0, dim_data['RMSE'], where=(dim_data['RMSE'] >= p75), alpha=0.5, color='red')
        ax.fill_between(dim_data['Frame'], 0, dim_data['RMSE'], where=(dim_data['RMSE'] <= p25), alpha=0.5, color='green')
        ax.fill_between(dim_data['Frame'], 0, dim_data['RMSE'], where=(dim_data['RMSE'] > p25) & (dim_data['RMSE'] < p75), alpha=0.5, color='yellow')
        
        ax.set_title(f"{joint_name} - {dim.upper()} Dimension")
        ax.set_xlabel('Frame')
        ax.set_ylabel('RMSE')
        ax.legend()
        
    plt.tight_layout()
    plt.show()

plot_error_trajectory(rmse_per_frame_df=rmse_per_frame, joint_name='left_heel', dimensions=['x', 'y', 'z'])
f = 2
from main import main
from RMSE_GUI import run_GUI
import numpy as np
from freemocap_utils.GUI_widgets.rmse_widgets.RMSE_calculator import calculate_rmse_dataframe, calculate_rmse_per_timepoint_per_dimension
from rmse_plots import calculate_rmse_per_frame, calculate_max_min_errors, plot_trajectory_with_error_shading
from marker_lists.mediapipe_markers import mediapipe_markers
from marker_lists.qualisys_markers import qualisys_markers

import pandas as pd
from pathlib import Path
def convert_3d_array_to_dataframe(data_3d_array:np.ndarray, data_marker_list:list):
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
        'Frame': frame_list,
        'Marker': marker_list,
        'x': x_list,
        'y': y_list,
        'z': z_list
    })

    return data_frame_marker_dim_dataframe

qualisys_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3\output_data\clipped_qualisys_skel_3d.npy"
freemocap_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\mediapipe_body_3d_xyz.npy"
freemocap_output_folder_path = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data")


freemocap_data = np.load(freemocap_data_path)
qualisys_data = np.load(qualisys_data_path)


freemocap_data_transformed = main(freemocap_data=freemocap_data, qualisys_data=qualisys_data, representative_frame=800)
np.save(freemocap_output_folder_path/'mediapipe_body_3d_xyz_transformed.npy', freemocap_data_transformed)


freemocap_df = convert_3d_array_to_dataframe(data_3d_array=freemocap_data_transformed, data_marker_list=mediapipe_markers)
qualisys_df = convert_3d_array_to_dataframe(data_3d_array=qualisys_data, data_marker_list=qualisys_markers)


# rmse_dataframe = calculate_rmse_dataframe(qualisys_data=qualisys_data, freemocap_data=freemocap_data_transformed)

rmse_per_frame = calculate_rmse_per_frame(freemocap_df=freemocap_df, qualisys_df=qualisys_df)
# plot_error_heatmap(rmse_per_frame)

# min_max_errors = calculate_max_min_errors(qualisys_data=qualisys_data, freemocap_data=freemocap_data_transformed, qualisys_indices=qualisys_markers, mediapipe_indices=mediapipe_markers)


# Create a function for plotting

plot_trajectory_with_error_shading(freemocap_df=freemocap_df, qualisys_df=qualisys_df, rmse_per_frame_df=rmse_per_frame, joint_name='left_heel', dimensions=['x', 'y', 'z'])
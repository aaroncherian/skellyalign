from data_utils.load_and_process_data import load_and_process_data
from error_calculations.get_error_metrics import get_error_metrics
from pathlib import Path


import matplotlib.pyplot as plt



qualisys_data_path = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3\output_data\clipped_qualisys_skel_3d.npy")
freemocap_data_path = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\mediapipe_body_3d_xyz_transformed.npy")

dataframe_of_3d_data = load_and_process_data(path_to_freemocap_data=freemocap_data_path, path_to_qualisys_data=qualisys_data_path)

error_metrics_dict = get_error_metrics(dataframe_of_3d_data=dataframe_of_3d_data)

error_metrics_dict['absolute_error_dataframe'].to_csv(Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\absolute_error_dataframe.csv"))
error_metrics_dict['rmse_dataframe'].to_csv(Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\rmse_dataframe.csv"))
f = 2 


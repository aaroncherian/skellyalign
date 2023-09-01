from main import main
from RMSE_GUI import run_GUI
import numpy as np

qualisys_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3\output_data\clipped_qualisys_skel_3d.npy"
freemocap_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\mediapipe_body_3d_xyz.npy"

freemocap_data = np.load(freemocap_data_path)
qualisys_data = np.load(qualisys_data_path)


freemocap_data_transformed = main(freemocap_data=freemocap_data, qualisys_data=qualisys_data, representative_frame=800)

run_GUI(qualisys_data=qualisys_data, freemocap_data=freemocap_data_transformed)

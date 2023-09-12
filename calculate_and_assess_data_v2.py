from data_utils.dataframe_builder import DataFrameBuilder
from marker_lists.mediapipe_markers import mediapipe_markers
from marker_lists.qualisys_markers import qualisys_markers

from pathlib import Path
import pandas as pd

def combine_3d_dataframes(dataframe_A, dataframe_B):
    return pd.concat([dataframe_A, dataframe_B], ignore_index=True)

qualisys_data_path = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3\output_data\clipped_qualisys_skel_3d.npy")
freemocap_data_path = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\mediapipe_body_3d_xyz.npy")

freemocap_builder = DataFrameBuilder(path_to_data=freemocap_data_path, marker_list=mediapipe_markers)
freemocap_data_dict = (freemocap_builder
                  .load_data()
                  .extract_common_markers()
                  .convert_to_dataframe()
                  .build())
freemocap_dataframe = freemocap_data_dict['dataframe_of_3d_data']

qualisys_builder = DataFrameBuilder(path_to_data=qualisys_data_path, marker_list=qualisys_markers)
qualisys_data_dict = (qualisys_builder
                    .load_data()
                    .extract_common_markers()
                    .convert_to_dataframe()
                    .build())
qualisys_dataframe = qualisys_data_dict['dataframe_of_3d_data']

freemocap_dataframe['system'] = 'freemocap'
qualisys_dataframe['system'] = 'qualisys'

combined_3d_dataframe = combine_3d_dataframes(dataframe_A=freemocap_dataframe, dataframe_B=qualisys_dataframe)

f =2 



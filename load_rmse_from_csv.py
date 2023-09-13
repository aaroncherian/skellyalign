

import pandas as pd

# Load the DataFrame
csv_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\rmse_dataframe.csv"
df = pd.read_csv(csv_path).reset_index(drop=True)

f = 2
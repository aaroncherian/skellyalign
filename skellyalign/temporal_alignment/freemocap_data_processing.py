import numpy as np
import pandas as pd

def create_freemocap_unix_timestamps(csv_path):
    df = pd.read_csv(csv_path)
    df.replace(-1, float('nan'), inplace=True)
    mean_timestamps = df.iloc[:, 2:].mean(axis=1, skipna=True)
    time_diff = np.diff(mean_timestamps)
    framerate = 1 / np.nanmean(time_diff)
    print(f"Calculated FreeMoCap framerate: {framerate}")
    return mean_timestamps, framerate


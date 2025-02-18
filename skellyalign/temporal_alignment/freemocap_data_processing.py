import numpy as np
import pandas as pd
from pathlib import Path
from skellyalign.temporal_alignment.utils.rotation import run_skellyforge_rotation

def create_freemocap_unix_timestamps(csv_path):
    df = pd.read_csv(csv_path)
    df.replace(-1, float('nan'), inplace=True)
    mean_timestamps = df.iloc[:, 2:].mean(axis=1, skipna=True)
    time_diff = np.diff(mean_timestamps)
    framerate = 1 / np.nanmean(time_diff)
    print(f"Calculated FreeMoCap framerate: {framerate}")
    return mean_timestamps, framerate

class FreeMoCapData:
    def __init__(self, file_path:Path):
        self.file_path = file_path

    def load_data(self):
        self.data = np.load(self.file_path)
        return self.data
    
    def rotate_data(self, landmark_names):
        self.rotated_data = run_skellyforge_rotation(
            raw_skeleton_data=self.data, 
            landmark_names=landmark_names)
        return self.rotated_data
from dataclasses import dataclass
import pandas as pd
import numpy as np

from  skellyalign.temporal_alignment.run_skellyforge_rotation import run_skellyforge_rotation
@dataclass
class QualisysResampler:
    joint_centers_with_unix_timestamps: pd.DataFrame
    freemocap_timestamps: pd.Series
    joint_center_names: list

    def __post_init__(self):
        self.resampled_qualisys_data = self.resample_qualisys_data(self.joint_centers_with_unix_timestamps, self.freemocap_timestamps)

    def resample_qualisys_data(self, qualisys_df, freemocap_timestamps):
        """
        Resample Qualisys data to match FreeMoCap timestamps using bin averaging.
        
        Parameters:
        -----------
        qualisys_df : pandas.DataFrame
            DataFrame with Frame, Time, unix_timestamps and data columns
        freemocap_timestamps : pandas.Series
            Target timestamps to resample to
            
        Returns:
        --------
        pandas.DataFrame
            Resampled data matching freemocap timestamps
        """

        if isinstance(freemocap_timestamps, pd.Series):
            freemocap_timestamps = freemocap_timestamps.to_numpy()
        
        bins = np.append(freemocap_timestamps, freemocap_timestamps[-1] + 
                    (freemocap_timestamps[-1] - freemocap_timestamps[-2]))
    
        # Assign each row to a bin (-1 means it's after the last timestamp)
        qualisys_df['bin'] = pd.cut(qualisys_df['unix_timestamps'], 
                                bins=bins, 
                                labels=range(len(freemocap_timestamps)),
                                include_lowest=True)
        
        # Group by bin and calculate mean
        # Note: dropna=False keeps bins that might be empty
        resampled = qualisys_df.groupby('bin', observed=True).mean(numeric_only=True)
        
        # Handle the last timestamp like the original
        if resampled.index[-1] == len(freemocap_timestamps) - 1:
            last_timestamp = freemocap_timestamps[-1]
            last_frame_data = qualisys_df[qualisys_df['unix_timestamps'] >= last_timestamp].iloc[0]
            resampled.iloc[-1] = last_frame_data[resampled.columns]
        
        resampled_qualisys_data = resampled.reset_index(drop=True)
        
        return resampled_qualisys_data
    
    def _create_marker_array(self) -> np.ndarray:
        """Convert marker data to a NumPy array of shape (frames, markers, 3)."""
        if not hasattr(self, 'resampled_qualisys_data'):
            raise AttributeError("No data available to resample. Resample Qualisys data first.")
        marker_data = self._extract_marker_data(self.resampled_qualisys_data)
        num_frames = len(marker_data)
        num_markers = int(len(marker_data.columns) / 3)

        return marker_data.to_numpy().reshape(num_frames, num_markers, 3)
    
    def _extract_marker_data(self, marker_and_timestamp_dataframe) -> pd.DataFrame:
        """Extract only marker data columns."""
        columns_of_interest = marker_and_timestamp_dataframe.columns[
            ~marker_and_timestamp_dataframe.columns.str.contains(r'^(?:Frame|Time|unix_timestamps|Unnamed)', regex=True)
        ]
        return marker_and_timestamp_dataframe[columns_of_interest]
    
    @property
    def resampled_marker_array(self):
        return self._create_marker_array()
    
    @property 
    def rotated_resampled_marker_array(self):
        return run_skellyforge_rotation(self.resampled_marker_array, self.joint_center_names)
    

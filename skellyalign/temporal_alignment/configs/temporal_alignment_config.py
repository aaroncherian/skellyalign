from dataclasses import dataclass
import numpy as np



@dataclass
class LagCorrectionSystemComponent:
    joint_center_array: np.ndarray
    list_of_joint_center_names: list

    def __post_init__(self):
        if len(self.joint_center_array) != len(self.list_of_joint_center_names):
            raise ValueError(f"Number of joint centers: {len(self.joint_center_array)} must match the number of joint center names: {len(self.list_of_joint_center_names)}")
        
@dataclass
class LagCorrector:
    freemocap_component: LagCorrectionSystemComponent
    qualisys_component: LagCorrectionSystemComponent
    framerate: float

    def get_common_joint_center_names(self, freemocap_joint_center_names, qualisys_joint_center_names):
        return list(set(freemocap_joint_center_names) & set(qualisys_joint_center_names))
    
    def calculate_lag_for_common_joints(self, 
                                        freemocap_joint_centers_array:np.ndarray, 
                                        qualisys_joint_centers_array:np.ndarray, 
                                        freemoocap_joint_centers_names:list, 
                                        qualisys_joint_centers_names:list, 
                                        common_joint_centers:list):
        
        optimal_lag_list = []

        for joint_center in common_joint_centers:
            qualisys_joint_idx = qualisys_joint_centers_names.index(joint_center)
            freemocap_joint_idx = freemoocap_joint_centers_names.index(joint_center)

            qualisys_joint_data = qualisys_joint_centers_array[:,qualisys_joint_idx,:]
            freemocap_joint_data = freemocap_joint_centers_array[:,freemocap_joint_idx,:]

            lag = self.calculate_lag(freemocap_joint_data, qualisys_joint_data)
            print(f"Lags per dimension for joint center {joint_center}: {lag}")

        optimal_lag_list.append(lag)

    def calculate_lag(self,
                      freemocap_joint_centers_array:np.ndarray,
                      qualisys_joint_centers_array:np.ndarray):
        """
        Calculate the optimal lag for a single marker across all three dimensions (X, Y, Z).

        Parameters:
            freemocap_joint_centers_array (np.ndarray): FreeMoCap data of shape (frames, 1, 3) for a single marker.
            qualisys_joint_centers_array (np.ndarray): Qualisys data of shape (frames, 1, 3) for a single marker.

        Returns:
            np.ndarray: Optimal lags for each dimension (X, Y, Z).
        """

        optimal_lags = []
        for dim in range(3):  # Loop over X, Y, Z
            freemocap_dim = freemocap_joint_centers_array[:, dim]
            qualisys_dim = qualisys_joint_centers_array[:, dim]

            # Ensure the signals are the same length
            min_length = min(len(freemocap_dim), len(qualisys_dim))
            freemocap_dim = freemocap_dim[:min_length]
            qualisys_dim = qualisys_dim[:min_length]

            # Normalize the data
            normalized_freemocap = self.normalize(freemocap_dim)
            normalized_qualisys = self.normalize(qualisys_dim)

            # Compute cross-correlation
            cross_corr = np.correlate(normalized_freemocap, normalized_qualisys, mode='full')

            # Find the lag that maximizes the cross-correlation
            optimal_lag = np.argmax(cross_corr) - (len(normalized_qualisys) - 1)
            optimal_lags.append(optimal_lag)
        
        optimal_lags = np.array(optimal_lags)
        return optimal_lag


    def normalize(self, 
                  signal: np.ndarray) -> np.ndarray:
            """
            Normalize a signal to have zero mean and unit variance.
            
            Parameters:
                signal (np.ndarray): The signal to normalize.

            Returns:
                np.ndarray: The normalized signal.
            """
            return (signal - signal.mean()) / signal.std()
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import random


class MotionCaptureData:
    def __init__(self, data_3d_array, marker_list):
        self.original_data_3d_array = data_3d_array
        self.marker_list = marker_list
        self.extracted_3d_array = None

        if len(marker_list) != data_3d_array.shape[1]:
            raise ValueError(f"Number of markers in marker_list ({len(marker_list)}) does not match the number of markers in data_3d_array ({data_3d_array.shape[1]}).")

    def _extract_specific_markers(self, data_marker_dimension, list_of_markers, markers_to_extract):
        indices = [list_of_markers.index(marker) for marker in markers_to_extract]
        return data_marker_dimension[:, indices, :]

    def extract_common_markers(self, markers_to_extract):
        """
        Extract markers from a specified list 
        Returns:
        self
        """
        self.markers_to_extract = markers_to_extract

        if self.original_data_3d_array is None:
            raise ValueError(f"data_3d_array is None. You must provide data.")

        self.extracted_3d_array = self._extract_specific_markers(
            data_marker_dimension=self.original_data_3d_array,
            list_of_markers=self.marker_list,
            markers_to_extract=markers_to_extract)
        return self

def optimize_transformation_least_squares(transformation_matrix_guess, data_to_transform, reference_data):
    tx, ty, tz, rx, ry, rz, s = transformation_matrix_guess
    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    transformed_data = s * rotation.apply(data_to_transform) + [tx, ty, tz]
    residuals = reference_data - transformed_data
    return residuals.flatten()

def run_least_squares_optimization(data_to_transform, reference_data, initial_guess=[0,0,0,0,0,0,1]):
    result = least_squares(optimize_transformation_least_squares, initial_guess, args=(data_to_transform, reference_data), gtol=1e-10, verbose=2)
    return result.x

def apply_transformation(transformation_matrix, data):
    tx, ty, tz, rx, ry, rz, s = transformation_matrix
    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    transformed_data = s * rotation.apply(data.reshape(-1, 3)) + np.array([tx, ty, tz])
    return transformed_data.reshape(data.shape)


def align_freemocap_and_qualisys_data_ransac(freemocap_data, qualisys_data, frames_to_sample=100, initial_guess=[0,0,0,0,0,0,1], max_iterations=1000, inlier_threshold=0.5):
    if freemocap_data.shape[1] != qualisys_data.shape[1]:
        raise ValueError("The number of markers in freemocap_data and qualisys_data must be the same.")
    
    num_frames = freemocap_data.shape[0]
    all_frames = list(range(num_frames))
    
    best_inliers = []
    best_transformation_matrix = None

    for iteration in range(max_iterations):
        # Randomly sample frames
        sampled_frames = random.sample(all_frames, frames_to_sample)
        
        # Prepare data for least squares optimization
        sampled_freemocap = freemocap_data[sampled_frames, :, :]
        sampled_qualisys = qualisys_data[sampled_frames, :, :]
        
        # Flatten the data for optimization
        flattened_freemocap = sampled_freemocap.reshape(-1, 3)
        flattened_qualisys = sampled_qualisys.reshape(-1, 3)
        
        # Fit transformation matrix using least squares
        transformation_matrix = run_least_squares_optimization(
            data_to_transform=flattened_freemocap, 
            reference_data=flattened_qualisys,
            initial_guess=initial_guess
        )
        
        # Apply the transformation to the entire dataset
        transformed_freemocap_data = apply_transformation(transformation_matrix, freemocap_data)
        
        # Calculate the alignment error for each frame
        errors = np.linalg.norm(qualisys_data - transformed_freemocap_data, axis=2).mean(axis=1)
        
        # Determine inliers based on the error threshold
        inliers = np.where(errors < inlier_threshold)[0]
        
        # Update the best model if the current model has more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_transformation_matrix = transformation_matrix
    
    # Apply the best transformation matrix to align the entire dataset
    if best_transformation_matrix is not None:
        aligned_freemocap_data = apply_transformation(best_transformation_matrix, freemocap_data)
        return aligned_freemocap_data
    else:
        raise ValueError("RANSAC failed to find a valid transformation.")


if __name__ == "__main__":
    from pathlib import Path

    from MDN_validation_marker_set import qualisys_markers, markers_to_extract, mediapipe_markers

    qualisys_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2\qualisys_data\qualisys_joint_centers_3d_xyz.npy"
    freemocap_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2\mediapipe_output_data\mediapipe_body_3d_xyz.npy"
    freemocap_output_folder_path = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data")

    freemocap_data = np.load(freemocap_data_path)
    qualisys_data = np.load(qualisys_data_path)


    # Extract common markers
    qualisys_mocap = MotionCaptureData(data_3d_array=qualisys_data, marker_list=qualisys_markers)
    freemocap_mocap = MotionCaptureData(data_3d_array=freemocap_data, marker_list=mediapipe_markers)
    
    qualisys_mocap.extract_common_markers(markers_to_extract=markers_to_extract)
    freemocap_mocap.extract_common_markers(markers_to_extract=markers_to_extract)

    aligned_freemocap_data = align_freemocap_and_qualisys_data_ransac(
        freemocap_data=freemocap_mocap.extracted_3d_array, 
        qualisys_data=qualisys_mocap.extracted_3d_array, 
        frames_to_sample=5, 
        max_iterations=100, 
        inlier_threshold=60
    )

    from scatter_3d import plot_3d_scatter
    plot_3d_scatter(freemocap_data=aligned_freemocap_data, qualisys_data=qualisys_data)
    # np.save(freemocap_output_folder_path/'mediapipe_body_3d_xyz_aligned.npy', aligned_freemocap_data)

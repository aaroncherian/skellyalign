# from skellyalign.models.alignment_config import RecordingConfig
from skellyalign.models.alignment_config import SpatialAlignmentConfig
from skellyalign.utilities.data_handlers import DataLoader, DataProcessor
from skellyalign.align_things import get_best_transformation_matrix_ransac, apply_transformation
from skellyalign.plots.scatter_3d import plot_3d_scatter

from skellymodels.skeleton_models.skeleton import Skeleton
from typing import List

import numpy as np

# def run_ransac_alignment_from_recording_config(recording_config:RecordingConfig):

#     freemocap_dataloader = DataLoader(path=recording_config.path_to_freemocap_output_data)
#     freemocap_data_processor = DataProcessor(data=freemocap_dataloader.data_3d, marker_list=recording_config.freemocap_markers, markers_for_alignment=recording_config.markers_for_alignment)
#     freemocap_extracted = freemocap_data_processor.extracted_data_3d

#     qualisys_dataloader = DataLoader(path=recording_config.path_to_qualisys_output_data)
#     qualisys_data_processor = DataProcessor(data=qualisys_dataloader.data_3d, marker_list=recording_config.qualisys_markers, markers_for_alignment=recording_config.markers_for_alignment)
#     qualisys_extracted = qualisys_data_processor.extracted_data_3d

#     best_transformation_matrix = get_best_transformation_matrix_ransac(freemocap_data = freemocap_extracted, qualisys_data = qualisys_extracted, frames_to_sample=recording_config.frames_to_sample, max_iterations=recording_config.max_iterations, inlier_threshold=recording_config.inlier_threshold)
#     aligned_freemocap_data = apply_transformation(best_transformation_matrix, freemocap_dataloader.data_3d)
    
#     return aligned_freemocap_data
#     # plot_3d_scatter(freemocap_data=aligned_freemocap_data, qualisys_data=qualisys_dataloader.data_3d)

def validate_marker_presence(freemocap_model: Skeleton, qualisys_model: Skeleton, markers_for_alignment: List[str]):
    """
    Validates the presence of alignment markers in both FreeMoCap and Qualisys skeleton models.

    This function checks whether all the markers specified for alignment are present in the marker lists
    of both the FreeMoCap and Qualisys skeleton models. If any markers are missing, it raises a ValueError
    with a descriptive message indicating which markers are missing in which model.

    Parameters:
    ----------
    freemocap_model : Skeleton
        The FreeMoCap skeleton model containing the marker names.
    qualisys_model : Skeleton
        The Qualisys skeleton model containing the marker names.
    markers_for_alignment : List[str]
        A list of marker names that are required for alignment.

    Raises:
    ------
    ValueError:
        If any markers specified for alignment are missing in either the FreeMoCap or Qualisys marker lists.
    """
    
    # Convert marker names from the skeleton models to sets for efficient comparison
    freemocap_markers = set(freemocap_model.marker_names)
    qualisys_markers = set(qualisys_model.marker_names)

    # Determine which alignment markers are missing in the FreeMoCap model
    missing_in_freemocap = set(markers_for_alignment) - freemocap_markers

    # Determine which alignment markers are missing in the Qualisys model
    missing_in_qualisys = set(markers_for_alignment) - qualisys_markers

    # Raise an error if any alignment markers are missing in the FreeMoCap model
    if missing_in_freemocap:
        raise ValueError(f"These markers for alignment were not found in FreeMoCap markers: {missing_in_freemocap}")

    # Raise an error if any alignment markers are missing in the Qualisys model
    if missing_in_qualisys:
        raise ValueError(f"These markers for alignment were not found in Qualisys markers: {missing_in_qualisys}")

def run_ransac_spatial_alignment(alignment_config: SpatialAlignmentConfig):
    """
    Runs the RANSAC spatial alignment process using the provided configuration.

    Parameters:
    ----------
    alignment_config : SpatialAlignmentConfig
        The configuration for the alignment process.

    Returns:
    -------
    aligned_freemocap_data : np.ndarray
        The aligned FreeMoCap data.
    best_transformation_matrix : np.ndarray
        The best transformation matrix obtained from the RANSAC process.
    """
    freemocap_model = alignment_config.freemocap_skeleton_function()
    qualisys_model = alignment_config.qualisys_skeleton_function()

    validate_marker_presence(
        freemocap_model=freemocap_model,
        qualisys_model=qualisys_model,
        markers_for_alignment=alignment_config.markers_for_alignment
    )

    freemocap_data = np.load(alignment_config.path_to_freemocap_output_data)
    freemocap_model.integrate_freemocap_3d_data(freemocap_data)

    qualisys_data = np.load(alignment_config.path_to_qualisys_output_data)
    qualisys_model.integrate_freemocap_3d_data(qualisys_data)

    freemocap_data_handler = DataProcessor(
        data=freemocap_model.marker_data_as_numpy,
        marker_list=freemocap_model.marker_names,
        markers_for_alignment=alignment_config.markers_for_alignment
    )
    qualisys_data_handler = DataProcessor(
        data=qualisys_model.marker_data_as_numpy,
        marker_list=qualisys_model.marker_names,
        markers_for_alignment=alignment_config.markers_for_alignment
    )

    best_transformation_matrix = get_best_transformation_matrix_ransac(
        freemocap_data=freemocap_data_handler.extracted_data_3d,
        qualisys_data=qualisys_data_handler.extracted_data_3d,
        frames_to_sample=alignment_config.frames_to_sample,
        max_iterations=alignment_config.max_iterations,
        inlier_threshold=alignment_config.inlier_threshold
    )
    aligned_freemocap_data = apply_transformation(
        best_transformation_matrix, freemocap_model.marker_data_as_numpy
    )

    return aligned_freemocap_data, best_transformation_matrix
    

    f = 2






    # freemocap_data_processor = DataProcessor(data=alignment_config.freemocap_skeleton.marker_data_as_numpy, marker_list=alignment_config.freemocap_skeleton.marker_names, markers_for_alignment=alignment_config.markers_for_alignment)
    # freemocap_data_for_alignment = freemocap_data_processor.extracted_data_3d

    # qualisys_data_processor = DataProcessor(data=alignment_config.qualisys_skeleton.marker_data_as_numpy, marker_list=alignment_config.qualisys_skeleton.marker_names, markers_for_alignment=alignment_config.markers_for_alignment)
    # qualisys_data_for_alignment = qualisys_data_processor.extracted_data_3d

    # best_transformation_matrix = get_best_transformation_matrix_ransac(freemocap_data = freemocap_data_for_alignment, qualisys_data = qualisys_data_for_alignment, frames_to_sample=alignment_config.frames_to_sample, max_iterations=alignment_config.max_iterations, inlier_threshold=alignment_config.inlier_threshold)

    # aligned_freemocap_data = apply_transformation(transformation_matrix=best_transformation_matrix, data=alignment_config.freemocap_skeleton.marker_data_as_numpy)

    return None
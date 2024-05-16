from skellyalign.models.recording_config import RecordingConfig
from skellyalign.utilities.data_handlers import DataLoader, DataProcessor
from skellyalign.align_things import get_best_transformation_matrix_ransac, apply_transformation
from skellyalign.plots.scatter_3d import plot_3d_scatter

def run_ransac_alignment(recording_config:RecordingConfig):

    freemocap_dataloader = DataLoader(path=recording_config.path_to_freemocap_output_data)
    freemocap_data_processor = DataProcessor(data=freemocap_dataloader.data_3d, marker_list=recording_config.freemocap_markers, markers_for_alignment=recording_config.markers_for_alignment)
    freemocap_extracted = freemocap_data_processor.extracted_data_3d

    qualisys_dataloader = DataLoader(path=recording_config.path_to_qualisys_output_data)
    qualisys_data_processor = DataProcessor(data=qualisys_dataloader.data_3d, marker_list=recording_config.qualisys_markers, markers_for_alignment=recording_config.markers_for_alignment)
    qualisys_extracted = qualisys_data_processor.extracted_data_3d

    best_transformation_matrix = get_best_transformation_matrix_ransac(freemocap_data = freemocap_extracted, qualisys_data = qualisys_extracted, frames_to_sample=recording_config.frames_to_sample, max_iterations=recording_config.max_iterations, inlier_threshold=recording_config.inlier_threshold)
    aligned_freemocap_data = apply_transformation(best_transformation_matrix, freemocap_dataloader.data_3d)
    
    return aligned_freemocap_data
    # plot_3d_scatter(freemocap_data=aligned_freemocap_data, qualisys_data=qualisys_dataloader.data_3d)

    f = 2   




from skellyalign.models.recording_config import RecordingConfig
from MDN_validation_marker_set import markers_to_extract, qualisys_markers, mediapipe_markers


sample_recording_config = RecordingConfig(
    path_to_recording="path/to/recording",
    path_to_freemocap_output_data="path/to/freemocap_output_data",
    path_to_qualisys_output_data="path/to/qualisys_output_data",
    freemocap_markers=mediapipe_markers,
    qualisys_markers=qualisys_markers,
    markers_for_alignment=markers_to_extract,
    frames_to_sample=20,
    max_iterations=20,
    inlier_threshold=50
)


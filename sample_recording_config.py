from skellyalign.models.recording_config import RecordingConfig
from MDN_validation_marker_set import markers_to_extract, qualisys_nih_markers, mediapipe_markers
from pathlib import Path


path_to_recording = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3")


sample_recording_config = RecordingConfig(
    path_to_recording=path_to_recording,
    path_to_freemocap_output_data = path_to_recording/'output_data'/'mediapipe_body_3d_xyz.npy',
    path_to_qualisys_output_data = path_to_recording/'qualisys'/ 'clipped_qualisys_skel_3d.npy',
    freemocap_markers=mediapipe_markers,
    qualisys_markers=qualisys_nih_markers,
    markers_for_alignment=markers_to_extract,
    frames_to_sample=20,
    max_iterations=20,
    inlier_threshold=50
)


from skellyalign.run_alignment import main
main(sample_recording_config)

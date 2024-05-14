from skellyalign.models.recording_config import RecordingConfig


sample_recording_config = RecordingConfig(
    path_to_recording="path/to/recording",
    path_to_freemocap_output_data="path/to/freemocap_output_data",
    path_to_qualisys_output_data="path/to/qualisys_output_data",
    freemocap_
    qualisys_marker_list=["marker1", "marker2"],
    markers_to_compare_list=["marker1", "marker2"],
    frame_for_comparison=0,
)


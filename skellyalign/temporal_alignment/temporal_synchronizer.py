
from skellyalign.temporal_alignment.configs.recording_config import Recording

from typing import List


from skellyalign.temporal_alignment.freemocap_data_processing import create_freemocap_unix_timestamps, FreeMoCapData
from skellyalign.temporal_alignment.synchronizing.lag_calculation import LagCalculatorComponent, LagCalculator
from skellyalign.temporal_alignment.qualisys_data_processing import QualisysMarkerData, QualisysJointCenterData, DataResampler
 
class TemporalSyncManager:
    def __init__(self, recording_config: Recording):
        self.recording_config = recording_config
    
    def _validate_required_metadata(self, component_name: str, required_keys: List[str]):
        component = self.recording_config.components[component_name]
        for key in required_keys:
            if not component.metadata.has(key):
                raise ValueError(f"Missing required metadata '{key}' for component {component_name}")

    def run(self):
        self._process_freemocap_data()
        self._load_freemocap_timestamps()
        self._process_qualisys_data()

        qualisys_component = self._create_qualisys_component(lag_in_seconds=0)
        initial_lag = self._calculate_lag(qualisys_component)
        
        corrected_qualisys_component = self._create_qualisys_component(lag_in_seconds=initial_lag)
        final_lag = self._calculate_lag(corrected_qualisys_component)

        print('Initial lag:', initial_lag)
        print('Final lag:', final_lag)
        ##this is for synchronizing the original non-joint center marker data as well so I can use it for trc creation 02/18/25
        resampler = DataResampler(self.qualisys_marker_data_holder.as_dataframe_with_unix_timestamps(lag_seconds=initial_lag), self.freemocap_timestamps)
        resampler.resample()

        marker_data_synced = resampler.as_dataframe

        return marker_data_synced

        f = 2 

    def _process_qualisys_data(self):

        self._validate_required_metadata('qualisys_exported_markers', ['joint_center_weights'])
        
        qualisys_marker_tsv_path = self.recording_config.get_component_file_path(
            'qualisys_exported_markers', 'markers')
        
        qualisys_marker_data_holder = QualisysMarkerData(qualisys_marker_tsv_path)
        qualisys_marker_data_holder.load_tsv()
        self.qualisys_marker_data_holder = qualisys_marker_data_holder

        self.qualisys_joint_center_data_holder = QualisysJointCenterData(
            marker_data_holder=qualisys_marker_data_holder,
            weights=self.recording_config.components['qualisys_exported_markers'].metadata.get('joint_center_weights')
        )

    def _process_freemocap_data(self):
        self._validate_required_metadata('mediapipe', ['landmark_names'])
        joint_center_names = self.recording_config.components['mediapipe'].metadata.get('landmark_names')
        freemocap_data_handler = FreeMoCapData(
            file_path=self.recording_config.get_component_file_path('mediapipe', 'body')
        )

        freemocap_data_handler.load_data()
        rotated_freemocap_data = freemocap_data_handler.rotate_data(
            landmark_names=joint_center_names
        )

        self.freemocap_component = LagCalculatorComponent(
            joint_center_array=rotated_freemocap_data, 
            list_of_joint_center_names=joint_center_names)

    def _create_qualisys_component(self, lag_in_seconds:float = 0) -> LagCalculatorComponent: 
        self._validate_required_metadata('qualisys_exported_markers', ['joint_center_names'])
        joint_center_names = self.recording_config.components['qualisys_exported_markers'].metadata.get('joint_center_names')
        df = self.qualisys_joint_center_data_holder.as_dataframe_with_unix_timestamps(lag_seconds=lag_in_seconds)
        resampler = DataResampler(df, self.freemocap_timestamps)
        resampler.resample()
        return LagCalculatorComponent(
            joint_center_array=resampler.rotated_resampled_marker_array(joint_center_names),
            list_of_joint_center_names=joint_center_names
        )

    def _calculate_lag(self, qualisys_component: LagCalculatorComponent):
        lag_corrector = LagCalculator(
            freemocap_component=self.freemocap_component, 
            qualisys_component=qualisys_component, 
            framerate=self.framerate)
        
        lag_corrector.run()
        print('Median lag:', lag_corrector.median_lag)
        print('Lag in seconds:', lag_corrector.get_lag_in_seconds())
        return lag_corrector.get_lag_in_seconds()

    def _load_freemocap_timestamps(self):
        timestamps_path = self.recording_config.get_component_file_path(
            'freemocap_timestamps', 'timestamps'
        )
        self.freemocap_timestamps, self.framerate = create_freemocap_unix_timestamps(csv_path=timestamps_path)
        f = 2
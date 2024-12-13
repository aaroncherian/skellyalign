from dataclasses import dataclass, field
from typing import Union, Dict, Optional
from pathlib import Path


@dataclass
class Component:
    name: str
    files: Dict[str,str]
    base_folder: Optional[str] = None

    def __post_init__(self):
        if not self.files:
            raise ValueError(f"Component {self.name} must have at least one file specified")
        

@dataclass
class RecordingSession:
    recording_folder_path: Union[str, Path]
    components: Dict[str, Component] = field(default_factory=dict)

    def __post_init__(self):
        self.recording_folder_path = Path(self.recording_folder_path)
        if not self.recording_folder_path.exists():
            raise ValueError(f"Recording folder path does not exist: {self.recording_folder_path}")
    
    
    @property
    def output_path(self) -> Path:
        return self.recording_folder_path / 'output_data'
    
    def add_component(self, 
                     name: str, 
                     files: Dict[str, str], 
                     base_folder: Optional[str] = None):
        """
        Add a component to the session
        
        Args:
            name: Name of the component
            files: Dictionary mapping file keys to filenames
            base_folder: Optional subfolder within output_data where specific component files are stored
            
        Raises:
            ValueError: If any specified files don't exist or if files dict is empty
        """
        if not files:
            raise ValueError(f"Component {name} must have at least one file specified")
            
        if name in self.components:
            raise ValueError(f"Component {name} already exists")

        # Verify all files exist before creating component
        base_path = self.output_path / base_folder if base_folder else self.output_path
        missing_files = []
        
        for file_key, file in files.items():
            file_path = base_path / file
            if not file_path.exists():
                missing_files.append(f"{file_key}: {file_path}")
        
        if missing_files:
            raise ValueError(
                f"The following files for component {name} do not exist:\n" +
                "\n".join(missing_files)
            )

        self.components[name] = Component(name=name, files=files, base_folder=base_folder)

    def get_component_file_path(self, 
                                component_name:str, 
                                file_name:str) ->  Optional[Path]:
        """
        Get path to a specific component file
        
        Args:
            component_name: Name of the component
            file_name: Key for the file in the component's files dict

        Returns:
            Path to the file or None if component/file not found
        """
        if component_name not in self.components:
            print(f'Component {component_name} not found in component list {self.components.keys()}')
            return None
        
        component = self.components[component_name]

        if file_name not in component.files:
            print(f'File {file_name} not found in component {component_name} files {component.files.keys()}')
            return None
        
        base_path = self.output_path / component.base_folder if component.base_folder else self.output_path
        file_path = base_path / component.files[file_name]

        return file_path


session = RecordingSession(
    recording_folder_path=r"D:\2024-10-30_treadmill_pilot\processed_data\sesh_2024-10-30_15_45_14_mdn_gait_7_exposure",
)

session.add_component(
    name = 'mediapipe',
    files = {
        'body': 'mediapipe_body_3d_xyz.npy'
    }
)

session.add_component(
    name = 'freemocap_timestamps',
    files = {
        'timestamps': 'unix_synced_timestamps.csv'
    },
)

session.add_component(
    name = 'qualisys_exported_markers',
    files = {
        'markers': 'qualisys_exported_markers.tsv'
    },
    base_folder = 'component_qualisys_original'
)

print(session.output_path)
print(session.get_component_file_path('mediapipe', 'body'))

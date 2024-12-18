from dataclasses import dataclass, field
from typing import Union, Dict, Optional, Any
from pathlib import Path

class ComponentMetadata:
    """Metadata for a given component"""
    def __init__(self):
        self._metadata = {}

    def add(self, key: str, value: Any) -> None:
        """
        Add metadata to the component
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
    
    def get(self, key: str, default = None) -> Any:
        """
        Get metadata value
        
        Args:
            key: Metadata key
            default: Default value to return if key not found
        
        Returns:
            Metadata value or default if key not found
        """
        value = self._metadata.get(key, default)
        if value is None:
            print(f"Warning: Metadata key '{key}' not found")
        return value
    
    def has(self, key: str) -> bool:
        """
        Check if metadata key exists
    
        Args:
            key: Metadata key
        
        Returns:
            True if key exists, False otherwise
        """
        return key in self._metadata
    
    def __getitem__(self, key: str) -> Any:
        return self._metadata[key]
        
    def __str__(self) -> str:
        return str(self._metadata)

@dataclass
class Component:
    name: str
    files: Dict[str,str]
    base_folder: Optional[str] = None
    metadata: ComponentMetadata = field(default_factory=ComponentMetadata)


    def __post_init__(self):
        if not self.files:
            raise ValueError(f"Component {self.name} must have at least one file specified")
        
class Recording:  # Regular class
    def __init__(self, recording_folder_path: Union[str, Path]):
        self.recording_folder_path = Path(recording_folder_path)
        if not self.recording_folder_path.exists():
            raise ValueError(f"Recording folder path does not exist: {self.recording_folder_path}")
    
        self.components: Dict[str, Component] = {}
    
    @property
    def output_path(self) -> Path:
        return self.recording_folder_path / 'output_data'

    @property
    def output_path(self) -> Path:
        return self.recording_folder_path / 'output_data'
    
    def _validate_component_files(self, name: str, files: Dict[str, str], base_folder: Optional[str]) -> None:
        """
        Validate component files exist and component can be added
        
        Args:
            name: Component name
            files: Dictionary mapping file keys to filenames
            base_folder: Optional subfolder within output_data
            
        Raises:
            ValueError: If validation fails
        """
        if not files:
            raise ValueError(f"Component {name} must have at least one file specified")
            
        if name in self.components:
            raise ValueError(f"Component {name} already exists")

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

    def add_component(self, 
                     name: str, 
                     files: Dict[str, str], 
                     base_folder: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Component:
        """
        Add a component to the session
        
        Args:
            name: Name of the component
            files: Dictionary mapping file keys to filenames
            base_folder: Optional subfolder within output_data where specific component files are stored
            
        Raises:
            ValueError: If any specified files don't exist or if files dict is empty
        """
        self._validate_component_files(name, files, base_folder)
        # self.components[name] = Component(name=name, files=files, base_folder=base_folder)
        component = Component(name=name, files=files, base_folder=base_folder)

        if metadata:
            for key, value in metadata.items():
                component.metadata.add(key, value)
        
        self.components[name] = component
        return component

    def add_prepared_component(self, component: Component) -> None:
        """
        Add a pre-configured component
        
        Args:
            component: Pre-configured Component instance
            
        Raises:
            ValueError: If any specified files don't exist or if files dict is empty
        """
        self._validate_component_files(component.name, component.files, component.base_folder)
        self.components[component.name] = component

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

if __name__ == "__main__":

    recording = Recording(
        recording_folder_path=r"D:\2024-10-30_treadmill_pilot\processed_data\sesh_2024-10-30_15_45_14_mdn_gait_7_exposure",
    )

    
    mediapipe_metadata = {'name': 'mediapipe'}

    recording.add_component(
        name = 'mediapipe',
        files = {
            'body': 'mediapipe_body_3d_xyz.npy'
        },
        base_folder=None,
        metadata=mediapipe_metadata
    )

    print(recording.components)
    print(recording.get_component_file_path('mediapipe', 'body'))
    print(recording.components['mediapipe'].files)
    print(recording.components['mediapipe'].metadata)
    print(recording.components['mediapipe'].metadata.get('name'))
    print(recording.components['mediapipe'].metadata.get('non_existent_key'))
    print(recording.components['mediapipe'].metadata.has('name'))
from abc import ABC, abstractmethod
from typing import List
from datetime import datetime

class BaseAppConfig(ABC):
    """
    Main controller entry for dataset controllers and configurations
    """
    data_root: str
    available_datasets: list[str]

    @abstractmethod
    def get_time_frame(self, folder_name: int, idx: int) -> datetime:
        """Get time frame (image filename) in the specified folder"""
        pass

    @abstractmethod
    def get_available_datasets(self) -> List[str]:
        """Get list of available folder names from data/images"""
        pass
    
    @abstractmethod
    def get_images_in_folder(self, folder_name: str) -> List[str]:
        pass

    @abstractmethod
    def get_filename(self, folder_name: str, index: int) -> str:
        """Get the filename at the specified index in the given folder"""
        pass
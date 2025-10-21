"""
Application Configuration

Centralized configuration for the Thunderstorm Nowcasting Visualization app.
Manages data paths and available datasets from data/images folder.
"""

import os
from typing import List
from datetime import datetime

from ..source_config import BASE_FOLDER
from .base import BaseAppConfig

class WindyAppConfig(BaseAppConfig):
    """Main application configuration class"""
    data_root: str
    available_datasets: List[str]
    
    def __init__(self):
        # Get the data paths from map_setup
        self.data_root = BASE_FOLDER
        self.available_datasets = os.listdir(os.path.join(BASE_FOLDER))

    def get_time_frame(self, folder_name: int, idx: int) -> datetime:
        """Get time frame (image filename) in the specified folder"""
        # Get list of all folders
        images_name = self.get_images_in_folder(folder_name)
        if idx < 0 or idx >= len(images_name):
            return None

        time = datetime.strptime(images_name[idx].split(".")[0], "%Y%m%d-%H%M%S")
        return time

    def get_available_datasets(self) -> List[str]:
        """Get list of available folder names from data/images"""
        return self.available_datasets
    
    def get_images_in_folder(self, folder_name: str) -> List[str]:
        """Get list of image files in the specified folder"""
        images = [file for file in os.listdir(os.path.join(BASE_FOLDER, folder_name)) if file.endswith('.png') or file.endswith('.jpg')]
        images.sort(reverse=False)
        
        return images

    def get_filename(self, folder_name: str, index: int) -> str:
        """Get the filename at the specified index in the given folder"""
        images = self.get_images_in_folder(folder_name)
        if 0 <= index < len(images):
            return f"{folder_name}/{images[index]}"
        return ""
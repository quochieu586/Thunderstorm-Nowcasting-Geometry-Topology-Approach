"""
Application Configuration

Centralized configuration for the Thunderstorm Nowcasting Visualization app.
Manages data paths and available datasets from data/images folder.
"""

import os
from typing import List
from datetime import datetime

from .source_config import BASE_FOLDER, IMAGES_FOLDER, IDENTIFICATION_METHODS, PRECIPITATION_MODELS

class ProcessingConfig:
    """Configuration for storm identification and processing"""
    
    def __init__(self):
        self.default_threshold: int = 35
        self.default_filter_area: int = 20
        self.available_thresholds: List[int] = [20, 25, 30, 35, 40, 45, 50]
        self.available_filter_areas: List[int] = [5, 10, 15, 20, 25, 30, 40, 50]


class WindyAppConfig:
    """Main application configuration class"""
    
    def __init__(self):
        # Get the data paths from map_setup
        self.data_root = BASE_FOLDER
        self.available_folders = IMAGES_FOLDER
        self.processing = ProcessingConfig()
        
        # Available identification methods
        self.identification_methods = IDENTIFICATION_METHODS
        self.precipitation_models = PRECIPITATION_MODELS

    def get_time_frame(self, folder_name: int, idx: int) -> datetime:
        """Get time frame (image filename) in the specified folder"""
        # Get list of all folders
        images_name = self.get_images_in_folder(folder_name)
        if idx < 0 or idx >= len(images_name):
            return None

        time = datetime.strptime(images_name[idx].split(".")[0], "%Y%m%d-%H%M%S")
        return time

    def get_available_folders(self) -> List[str]:
        """Get list of available folder names from data/images"""
        return self.available_folders
    
    def get_images_in_folder(self, folder_name: str) -> List[str]:
        """Get list of image files in the specified folder"""
        images = [file for file in os.listdir(os.path.join(BASE_FOLDER, folder_name)) if file.endswith('.png') or file.endswith('.jpg')]
        images.sort(reverse=False)
        
        return images
    
    def get_identification_methods(self) -> List[str]:
        """Get list of available identification method names"""
        return list(self.identification_methods.keys())

    def get_filename(self, folder_name: str, index: int) -> str:
        """Get the filename at the specified index in the given folder"""
        images = self.get_images_in_folder(folder_name)
        if 0 <= index < len(images):
            return f"{folder_name}/{images[index]}"
        return ""
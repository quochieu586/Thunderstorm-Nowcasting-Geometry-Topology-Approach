"""
Application Configuration

Centralized configuration for the Thunderstorm Nowcasting Visualization app.
Manages data paths and available datasets from data/images folder.
"""

import os
from typing import List
from .source_config import BASE_FOLDER, IMAGES_FOLDER, get_all_images

class ProcessingConfig:
    """Configuration for storm identification and processing"""
    
    def __init__(self):
        self.default_threshold: int = 35
        self.default_filter_area: int = 20
        self.available_thresholds: List[int] = [20, 25, 30, 35, 40, 45, 50]
        self.available_filter_areas: List[int] = [5, 10, 15, 20, 25, 30, 40, 50]


class AppConfig:
    """Main application configuration class"""
    
    def __init__(self):
        # Get the data paths from map_setup
        self.data_root = BASE_FOLDER
        self.available_folders = IMAGES_FOLDER
        self.processing = ProcessingConfig()
        
        # Available identification methods
        self.identification_methods = {
            "Simple Contour": "SimpleContourIdentifier",
            "Hypothesis": "HypothesisIdentifier", 
            "Morphology": "MorphContourIdentifier",
            "Cluster": "ClusterIdentifier"
        }
    
    def get_available_folders(self) -> List[str]:
        """Get list of available folder names from data/images"""
        return self.available_folders
    
    def get_images_in_folder(self, folder_name: str) -> List[str]:
        """Get list of image files in the specified folder"""
        if folder_name in self.available_folders:
            return get_all_images(folder_name)
        return []
    
    def get_identification_methods(self) -> List[str]:
        """Get list of available identification method names"""
        return list(self.identification_methods.keys())
    
    def get_identification_class_name(self, method_name: str) -> str:
        """Get the class name for a given identification method"""
        return self.identification_methods.get(method_name, "SimpleContourIdentifier")

    def get_filename(self, folder_name: str, index: int) -> str:
        """Get the filename at the specified index in the given folder"""
        images = self.get_images_in_folder(folder_name)
        if 0 <= index < len(images):
            return f"{folder_name}/{images[index]}"
        return ""
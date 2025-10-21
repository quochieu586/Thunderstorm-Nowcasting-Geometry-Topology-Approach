"""
Data Processing Component

Handles loading and processing of radar data and storm identification.
Manages the interface between the UI and the core processing modules.
"""

import streamlit as st
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing import read_image, windy_preprocessing_pipeline, convert_contours_to_polygons

from src.cores.base import StormsMap, StormObject
from src.identification import BaseStormIdentifier

from src.models import BasePrecipitationModel, SimplePrecipitationModel

from app.config.source_config import IDENTIFICATION_METHODS, PRECIPITATION_MODELS
from app.config.app_config import BaseAppConfig

class DataProcessor:
    """
    Handles data loading, storm identification processing, and state management with cache mechanism for efficiency.

    Args:
        _image_cache: Cache for loaded images to avoid redundant I/O operations
        _storms_cache: Cache for identified storms to avoid redundant processing
        _models_cache: Cache for precipitation model instances per dataset. Where single model instance is used for single dataset.
    
    Cache reset: When parameters (method, dbz threshold, filter area) change, the storms and models cache are cleared.
    """

    def __init__(self, config: BaseAppConfig):
        self.config = config
        self.identifiers = IDENTIFICATION_METHODS
        self.precipitation_models = PRECIPITATION_MODELS
        
        # Cache for processed data
        self._image_cache: dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._storms_cache: dict[str, StormsMap] = {}
        self._models_cache: dict[str, BasePrecipitationModel] = {}
    
    # =============================================================================
    # Public Processing Methods
    # =============================================================================

    def _get_precipitation_models_object(self, dataset_name: str, method_name: str) -> BasePrecipitationModel:
        source_model = self.precipitation_models.get(method_name)

        if dataset_name not in self._models_cache:
            self._models_cache[dataset_name] = source_model.copy()

        return self._models_cache[dataset_name]

    def identify_storms(self, selected_folder: str, scan_index: int, precipitation_model: str, threshold: int, filter_area: int) -> tuple[np.ndarray, np.ndarray, StormsMap]:
        """
        Identify storms in a DBZ map (private method)
        
        Args:
            dbz_map: DBZ reflectivity map
            folder_name: Name of the folder containing images
            scan_index: Index of the scan
            precipitation_model: Name of precipitation model applying. This can be either: Simple Model, ETitan, AINT,... .
            threshold: DBZ threshold for storm identification
            filter_area: Minimum area filter for storms
            
        Returns:
            StormsMap object containing identified storms
        """
        cache_key = f"{selected_folder}_{scan_index}"

        print("Identifying storms with cache key:", cache_key)

        # Find in cache first
        original_image, dbz_map = self._image_cache.get(cache_key, (None, None))
        storms_map = self._storms_cache.get(cache_key, None)

        if original_image is None:
            original_image, dbz_map = self._load_image(selected_folder, scan_index)

        if storms_map is None:
            print(f"Cache missed for storms at key: {cache_key}, processing identification...")

            # Get the model object
            try:
                instance_model = self._get_precipitation_models_object(selected_folder, precipitation_model)
            except KeyError:
                raise ValueError(f"Precipitation model '{precipitation_model}' not found in available models.")

            # Processing the image: identify contours, track storms and update history of model
            time_frame = self.config.get_time_frame(selected_folder, scan_index)
            storms_map = instance_model.identify_storms(dbz_map, threshold=threshold, filter_area=filter_area, map_id=f"storm_{scan_index}", time_frame=time_frame)
            instance_model.processing_map(storms_map)            # Update tracking history

            # Cache the identified storms
            self._storms_cache[cache_key] = storms_map

        else:
            print(f"Cache hit for storms at key: {cache_key}, using cached data.")
            
        return original_image, dbz_map, storms_map
    
    def clear_storms_cache(self) -> None:
        """
        Clear the storms cache when identification parameters, dbz threshold or filter area change
        """
        print("Clearing storms and models cache...")
        self._storms_cache.clear()
        self._models_cache.clear()


    def get_cache_info(self) -> Dict[str, int]:
        """Get information about cached data"""
        return {
            "images_cached": len(self._image_cache),
            "storms_cached": len(self._storms_cache),
            "models_cached": len(self._models_cache),           # Single model for each dataset
        }
    
    # =============================================================================
    # Private Methods
    # =============================================================================

    def _load_image(self, folder_name: str, scan_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process a single scan image (private method)
        
        Args:
            folder_name: Name of the folder containing images
            scan_index: Index of the scan to load
            
        Returns:
            Tuple of (original_image, dbz_map)
        """
        
        # Check cache first
        cache_key = f"{folder_name}_{scan_index}"
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]
        
        # Get images in folder
        images = self.config.get_images_in_folder(folder_name)
        if not images:
            raise ValueError(f"No images found in folder {folder_name}")
        
        if scan_index >= len(images):
            raise IndexError(f"Scan index {scan_index} out of range for folder {folder_name}")
        
        # Construct file path
        image_filename = images[scan_index]
        file_path = Path(self.config.data_root) / folder_name / image_filename
        
        try:
            # Load image (assuming all images are PNG/JPG for now)
            original_image = read_image(str(file_path))
            dbz_map = windy_preprocessing_pipeline(original_image)
            
            # Cache the result
            result = (original_image, dbz_map)
            self._image_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            st.error(f"Error loading image {file_path}: {str(e)}")
            raise
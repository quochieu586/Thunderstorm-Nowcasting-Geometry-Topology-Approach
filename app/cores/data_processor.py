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

from src.cores.base import StormsMap
from src.identification import BaseStormIdentifier

from app.config import AppConfig, IDENTIFICATION_METHODS
from app.utils import StormWithMovements

class DataProcessor:
    """Handles data loading, storm identification processing, and state management"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.identifiers = IDENTIFICATION_METHODS
        
        # Cache for processed data
        self._image_cache: dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._storms_cache: dict[str, StormsMap] = {}
    
    # =============================================================================
    # Public Processing Methods
    # =============================================================================

    def identify_storms(self, selected_folder: str, scan_index: int, identification_method: str, threshold: int, filter_area: int) -> tuple[np.ndarray, np.ndarray, StormsMap]:
        """
        Identify storms in a DBZ map (private method)
        
        Args:
            dbz_map: DBZ reflectivity map
            folder_name: Name of the folder containing images
            scan_index: Index of the scan
            identification_method: Name of identification method to use
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

            # Get the appropriate identifier
            identifier = self.identifiers.get(identification_method)
            identifier.set_params(threshold=threshold, filter_area=filter_area)
            if identifier is None or not isinstance(identifier, BaseStormIdentifier):
                raise ValueError(f"Unknown identification method: {identification_method}")

            # Identify contours
            contours = identifier.identify_storm(dbz_map)
            
            # Convert contours to polygons
            polygons = convert_contours_to_polygons(contours)
            polygons = [pol for pol in polygons if pol.area >= filter_area]
            polygons = sorted(polygons, key=lambda x: x.area, reverse=True)
            
            # Create storm objects
            storms = [
                StormWithMovements(polygon, id=f"scan_{scan_index}_storm_{idx}") 
                for idx, polygon in enumerate(polygons)
            ]
            
            # Create storms map
            storms_map = StormsMap(storms, time_frame=datetime.now())
            
            # Cache the result
            self._storms_cache[cache_key] = storms_map
        else:
            print(f"Cache hit for storms at key: {cache_key}, using cached data.")
            
        return original_image, dbz_map, storms_map
    
    def clear_storms_cache(self) -> None:
        """
        Clear the storms cache when identification parameters, dbz threshold or filter area change
        """
        print("Clearing storms cache... . Before clearing, cache size:", len(self._storms_cache))
        self._storms_cache.clear()
        print("Detect changed in identification parameters, clearing storms cache ! After clearing, cache size:", len(self._storms_cache))

    def get_cache_info(self) -> Dict[str, int]:
        """Get information about cached data"""
        return {
            "images_cached": len(self._image_cache),
            "storms_cached": len(self._storms_cache)
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
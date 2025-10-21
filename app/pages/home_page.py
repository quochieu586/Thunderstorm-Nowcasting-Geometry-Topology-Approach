"""
Home Page

Main page for the Thunderstorm Nowcasting Visualization application.
Displays radar images with storm identification overlays and controls.
"""

import numpy as np

import streamlit as st
import sys
import os
from typing import Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.cores.base import StormsMap

from app.cores.data_processor import DataProcessor
from app.utils import draw_contours
from app.components.contours_window import ContourWindow

class HomePage:
    """Main page component for storm visualization"""
    def __init__(self, global_data_processor: DataProcessor):
        self.global_data_processor = global_data_processor
        self.contour_window = ContourWindow()
    
    def render(self) -> None:
        """Render the home page"""
        # Process and display current scan
        self._process_and_display_current_scan()

    def _process_and_display_current_scan(self) -> None:
        """Process and display the current scan with storm identification"""
        selected_folder = st.session_state.get('selected_folder')
        precipitation_model = st.session_state.get('precipitation_model', 'Simple Contour')
        threshold = st.session_state.get('dbz_threshold', 35)
        filter_area = st.session_state.get('filter_area', 20)
        scan_index = st.session_state.get('current_scan_index', 0)

        print(f"Rendering... . Processing scan with parameters: precipitation_model={precipitation_model}, threshold={threshold}, filter_area={filter_area}, selected_folder={selected_folder}, scan_index={scan_index}")

        try:
            with st.spinner("🔄 Processing radar data and identifying storms..."):
                # Process the current scan using DataProcessor's centralized state
                result = self.global_data_processor.identify_storms(selected_folder=selected_folder, scan_index=scan_index, precipitation_model=precipitation_model, threshold=threshold, filter_area=filter_area)

            if result is None:
                st.warning("No folder selected or no images available")
                return
    
            original_image, dbz_map, storms_map = result

            col1, col2 = st.columns(2)
            with col1:
                st.text("Number of storms identified: " + str(len(storms_map.storms)))
                st.text("File name: " + self.global_data_processor.config.get_filename(selected_folder, scan_index))
            with col2:
                st.text("Number of cache images: " + str(len(self.global_data_processor._image_cache)))
                st.text("Number of cache contours: " + str(len(self.global_data_processor._storms_cache)))

            # Display results
            self.contour_window.render(original_image, storms_map)
            
        except Exception as e:
            st.error(f"❌ Error processing scan: {str(e)}")
            st.exception(e)
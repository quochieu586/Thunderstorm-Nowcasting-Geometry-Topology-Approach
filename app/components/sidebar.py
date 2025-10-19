"""
Sidebar Component

Streamlit sidebar for dataset selection, identification method configuration,
and processing parameter controls.
"""

import streamlit as st
from typing import Optional

from config.app_config import AppConfig
from .data_processor import DataProcessor

class Sidebar:
    """Sidebar component for application controls"""
    def __init__(self, config: AppConfig, global_data_processor: DataProcessor):
        self.config = config
        self.global_data_processor = global_data_processor
        self.maximum_scans: Optional[int] = None  # Number of scans in the selected folder

    def render(self) -> None:
        """Render the complete sidebar"""
        with st.sidebar:
            st.header("Controls")
            
            # Dataset selection
            self._render_dataset_selection()
            st.divider()
            
            # Identification method selection
            self._render_identification_selection()
            st.divider()
            
            # Processing parameters
            self._render_processing_parameters()
            st.divider()

            # Scan index control
            self._render_scan_index_control()
    
    def _render_dataset_selection(self) -> None:
        """Render dataset selection controls"""
        st.subheader("Dataset Selection")
        
        available_folders = self.config.get_available_folders()
        
        if not available_folders:
            st.warning("No datasets found in data directory")
            return
        
        # Dataset selector
        st.selectbox(
            "Choose Location/Dataset:",
            options=available_folders,
            index=0,
            key="selected_folder",
            help="Select a geographical location or dataset to analyze",
        )
    
    def _render_identification_selection(self) -> None:
        """
        Render identification method selection
        """
        st.subheader("🔍 Identification Method")
        
        identification_methods = self.config.get_identification_methods()

        selected_method = st.selectbox(
            "Choose Identification Method:",
            options=identification_methods,
            index=0,
            key="identification_method",
            help="Select the storm identification algorithm to use",
            on_change=(lambda: self.global_data_processor.clear_storms_cache())
        )
        
        # Display method information
        method_descriptions = {
            "Simple Contour": "Identifies storms as contiguous pixels above DBZ threshold",
            "Hypothesis": "Uses dilation from maximum centers for subcell processing",
            "Morphology": "Applies morphological operations for storm detection",
            "Cluster": "Uses clustering algorithms to identify storm regions"
        }
        
        if selected_method in method_descriptions:
            st.caption(method_descriptions[selected_method])
    
    def _render_processing_parameters(self) -> None:
        """Render processing parameter controls"""
        st.subheader("⚙️ Processing Parameters")
        
        # DBZ Threshold
        st.select_slider(
            "DBZ Threshold:",
            options=self.config.processing.available_thresholds,
            value=self.config.processing.default_threshold,
            key="dbz_threshold",
            help="Minimum DBZ value to consider for storm identification",
            on_change=(lambda: self.global_data_processor.clear_storms_cache())
        )
        
        # Filter Area
        st.select_slider(
            "Minimum Area Filter:",
            options=self.config.processing.available_filter_areas,
            value=self.config.processing.default_filter_area,
            key="filter_area",
            help="Minimum area (pixels) to filter out small storm objects",
            on_change=(lambda: self.global_data_processor.clear_storms_cache())
        )

    def _navigate_scan(self, go_to: bool) -> None:
        """Navigate to a specific scan index"""
        if go_to:
            st.session_state.current_scan_index = max(0, st.session_state.current_scan_index - 1)
        else:
            st.session_state.current_scan_index = min(
                self.maximum_scans - 1,
                st.session_state.current_scan_index + 1
            )

    def _render_scan_index_control(self) -> None:
        """
        Render scan index control slider
        """
        self.maximum_scans = len(self.global_data_processor.config.get_images_in_folder(
            st.session_state.get('selected_folder')
        ))

        if self.maximum_scans is None or self.maximum_scans == 0:
            st.info("No scans available for the selected dataset")
            return
        
        st.slider(
            "Scan Index:",
            min_value=0,
            max_value=self.maximum_scans - 1,
            value=0,
            step=1,
            key="current_scan_index",
            help="Select the scan index to process",
        )

        # Bonus arrow buttons to navigate to previous/next scan
        col1, col2 = st.columns(2)
        with col1:
            st.button("← Previous Scan", on_click=lambda: self._navigate_scan(True))
        with col2:
            st.button("Next Scan →", on_click=lambda: self._navigate_scan(False))
"""
Sidebar Component

Streamlit sidebar for dataset selection, identification method configuration,
and processing parameter controls.
"""

import streamlit as st
from typing import Optional

from app.config.app_config import BaseAppConfig
from app.config.source_config import FILTER_AREA_OPTIONS, THRESHOLD_OPTIONS, PRECIPITATION_MODELS
from app.cores.data_processor import DataProcessor

class Sidebar:
    """Sidebar component for application controls"""
    def __init__(self, config: BaseAppConfig, global_data_processor: DataProcessor):
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

    def _reset_scanning(self, clear_cache: bool) -> None:
        """Reset scanning index to zero"""
        st.session_state.current_scan_index = 0
        if clear_cache:
            self.global_data_processor.clear_storms_cache()
    
    def _render_dataset_selection(self) -> None:
        """Render dataset selection controls"""
        st.subheader("Dataset Selection")
        
        available_datasets = self.config.get_available_datasets()
        
        if not available_datasets:
            st.warning("No datasets found in data directory")
            return
        
        # Dataset selector
        st.selectbox(
            "Choose Location/Dataset:",
            options=available_datasets,
            index=0,
            key="selected_folder",
            help="Select a geographical location or dataset to analyze",
            on_change=(lambda: self._reset_scanning(clear_cache=False))
        )
    
    def _render_identification_selection(self) -> None:
        """
        Render nowcasting method selection
        """
        st.subheader("🔍 Nowcasting Method")

        selected_method = st.selectbox(
            "Choose Precipitation Model:",
            options=PRECIPITATION_MODELS.keys(),
            index=0,
            key="precipitation_model",
            help="Select the storm precipitation model to use",
            on_change=(lambda: self.global_data_processor.clear_storms_cache())
        )

        method_descriptions = {
            "Simple Precipitation Model": "Identifies storms as contiguous pixels above DBZ threshold, use polar shape vectors & FFT for matching - tracking",
            "ETitan Precipitation Model": "Identifies storms using morphological operations: dilation and erosion"
        }
        
        if selected_method in method_descriptions:
            st.caption(method_descriptions[selected_method])
    
    def _render_processing_parameters(self) -> None:
        """Render processing parameter controls"""
        st.subheader("⚙️ Processing Parameters")
        
        # DBZ Threshold
        st.select_slider(
            "DBZ Threshold:",
            options=THRESHOLD_OPTIONS,
            value=THRESHOLD_OPTIONS[3],
            key="dbz_threshold",
            help="Minimum DBZ value to consider for storm identification",
            on_change=(lambda: self._reset_scanning(clear_cache=True))
        )
        
        # Filter Area
        st.select_slider(
            "Minimum Area Filter:",
            options=FILTER_AREA_OPTIONS,
            value=FILTER_AREA_OPTIONS[-1],
            key="filter_area",
            help="Minimum area (pixels) to filter out small storm objects",
            on_change=(lambda: self._reset_scanning(clear_cache=True))
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
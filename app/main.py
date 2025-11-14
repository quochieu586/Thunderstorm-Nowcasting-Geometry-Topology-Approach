"""
Thunderstorm Nowcasting Visualization App

A Streamlit application for visualizing storm identification and tracking
across multiple consecutive radar scans from different geographical locations.
"""

import streamlit as st
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pages.home_page import HomePage
from components.sidebar import Sidebar

from app.config.app_config import WindyAppConfig
from app.cores.data_processor import DataProcessor

def main():
    """Main application entry point"""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Thunderstorm Nowcasting Visualization",
        page_icon="⛈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize app configuration and data processor
    if 'app_config' not in st.session_state:
        st.session_state.app_config = WindyAppConfig()
        st.session_state.data_processor = DataProcessor(st.session_state.app_config)
    
    # Initialize home page (which creates data processor)
    home_page = HomePage(global_data_processor=st.session_state.data_processor)
    
    # Create sidebar with data processor reference
    sidebar = Sidebar(st.session_state.app_config, global_data_processor=st.session_state.data_processor)
    sidebar.render()
    
    # Main content area
    st.title("⛈️ Thunderstorm Nowcasting Visualization")
    st.markdown("---")
    
    # Render home page
    home_page.render()


if __name__ == "__main__":
    main()

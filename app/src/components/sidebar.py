import streamlit as st
import os
from app.src.config import BASE_IMAGE_DIR

def display_sidebar():
    """
    Displays the sidebar for data selection and parameter configuration.
    Returns the selected directory, image name, and processing parameters.
    """
    st.sidebar.header("Data and Parameters")

    try:
        subdirectories = sorted([d for d in os.listdir(BASE_IMAGE_DIR) if os.path.isdir(os.path.join(BASE_IMAGE_DIR, d))])
        if not subdirectories:
            st.error(f"No subdirectories found in directory: {BASE_IMAGE_DIR}")
            st.stop()
    except FileNotFoundError:
        st.error(f"Base image directory not found: {BASE_IMAGE_DIR}")
        st.stop()

    selected_subdir = st.sidebar.selectbox("Select a directory", subdirectories)
    image_dir = os.path.join(BASE_IMAGE_DIR, selected_subdir)

    try:
        files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        if not files:
            st.warning(f"No PNG images found in directory: {image_dir}")
            st.stop()
    except FileNotFoundError:
        st.error(f"Image directory not found: {image_dir}")
        st.stop()

    display_files = ["All"] + files
    selected_image_name = st.sidebar.selectbox("Select an image", display_files)

    dbz_threshold = st.sidebar.slider("dBZ Threshold", 10, 70, 20)
    contour_area_threshold = st.sidebar.slider("Contour Area Threshold", 0, 1000, 100)

    return image_dir, files, selected_image_name, dbz_threshold, contour_area_threshold
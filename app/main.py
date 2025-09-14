import streamlit as st
import os
import sys
import traceback
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components
from app.src.components import display_sidebar, display_images, display_persistence_diagram

# Import utils
from app.src.utils import get_contours, get_datetime_from_filename, load_image, create_indexed_contour_image
from utils.persistence_homology.persistence_diagram import PersistenceDiagram


# --- Page Config and State Initialization ---
st.set_page_config(layout="wide")
st.title("Thunderstorm Nowcasting - Persistence Homology")

if 'ph_diagrams' not in st.session_state:
    st.session_state.ph_diagrams = {}

# --- Sidebar ---
IMAGE_DIR, files, selected_image_name, dbz_threshold, contour_area_threshold = display_sidebar()


# --- Main Application Logic ---

if selected_image_name == "All":
    st.header("All Images Comparison")
    st.info("Showing all images in the selected directory. Thresholds are applied globally.")

    for image_file in files:
        st.markdown("---")
        image_path = os.path.join(IMAGE_DIR, image_file)
        image_datetime = get_datetime_from_filename(image_file)
        st.subheader(f"Timestamp: {image_datetime}")

        col1, col2 = st.columns(2)

        img_for_contours, contours = get_contours(image_path, dbz_threshold, contour_area_threshold)

        with col1:
            display_images(img_for_contours, contours)
        
        with col2:
            display_persistence_diagram(image_file, contours)

else:
    # --- SINGLE IMAGE VIEW ---
    selected_image_path = os.path.join(IMAGE_DIR, selected_image_name)

    # Display extracted datetime
    image_datetime = get_datetime_from_filename(selected_image_name)
    st.info(f"Image Timestamp: {image_datetime}")

    # --- Main Panel ---
    col1, col2 = st.columns(2)

    # --- Image Display ---
    with col1:
        st.subheader("Source Image")
        source_image = load_image(selected_image_path)
        if source_image is not None:
            st.image(source_image, caption="Original selected image.", width='stretch')

    with col2:
        st.subheader("Image with Contours")
        try:
            img_for_contours, contours = get_contours(selected_image_path, dbz_threshold, contour_area_threshold)
            if img_for_contours is not None:
                if contours:
                    img_with_contours = img_for_contours.copy()
                    cv2.drawContours(img_with_contours, contours, -1, (0, 0, 0), 2)
                    st.image(img_with_contours, caption=f"{len(contours)} contours detected.", width='stretch')
                else:
                    st.image(img_for_contours, caption="No contours found with current settings.", width='stretch')
        except Exception as e:
            st.error("Failed to process contours.")
            st.error(traceback.format_exc())

    # --- Indexed Contour Image Display ---
    st.subheader("Indexed Contour Map")
    try:
        # Re-use data from the previous step
        if 'img_for_contours' in locals() and img_for_contours is not None:
            if 'contours' in locals() and contours:
                indexed_contour_fig = create_indexed_contour_image(img_for_contours.shape, contours)
                plot_col, _ = st.columns(2)
                with plot_col:
                    st.pyplot(indexed_contour_fig)
            else:
                plot_col, _ = st.columns(2)
                with plot_col:
                    st.info("No contours to display in an indexed map.")
    except Exception as e:
        st.error("Failed to create indexed contour map.")
        st.error(traceback.format_exc())


    # --- Persistence Homology Section ---
    st.header("Persistence Homology Analysis")

    # Get contours for multiselect
    _, contours_for_ph = get_contours(selected_image_path, dbz_threshold, contour_area_threshold)
    contour_indices = list(range(len(contours_for_ph)))

    selected_indices = st.multiselect(
        "Select contour indices to include in Persistence Homology computation:",
        options=contour_indices,
        default=contour_indices  # Default to all contours
    )

    if st.button("Compute Persistence Homology"):
        if not selected_indices:
            st.warning("Please select at least one contour to compute persistence homology.")
        else:
            with st.spinner("Computing persistence diagram..."):
                try:
                    # Filter contours based on selection
                    selected_contours = [contours_for_ph[i] for i in selected_indices]
                    
                    data_cloud = np.concatenate([contour.squeeze() for contour in selected_contours])
                    res_pc = PersistenceDiagram.compute(data_cloud, maxdim=1)
                    
                    fig = plt.figure()
                    title = f'Persistence Diagram for contours {selected_indices} in {selected_image_name}'
                    PersistenceDiagram.plot_persistence_diagram(res_pc['dgms'], title=title)
                    
                    plot_col, _ = st.columns(2)
                    with plot_col:
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"An unexpected error occurred during computation: {e}")
                    st.error(traceback.format_exc())
    else:
        st.info("Select contours and click the button to generate the persistence homology diagram for the detected contours.")
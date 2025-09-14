import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.persistence_homology.persistence_diagram import PersistenceDiagram

def display_persistence_diagram(image_file, contours):
    """
    Handles the display and computation of the Persistence Homology diagram.
    """
    # Check if a diagram is already computed and stored
    if image_file in st.session_state.ph_diagrams:
        st.pyplot(st.session_state.ph_diagrams[image_file])
        if st.button("❌ Remove", key=f"remove_{image_file}"):
            del st.session_state.ph_diagrams[image_file]
            st.rerun()
    elif contours:
        contour_indices = list(range(len(contours)))
        selected_indices = st.multiselect(
            "Select contours:",
            options=contour_indices,
            default=contour_indices,
            key=f"multiselect_{image_file}"
        )

        if st.button("Compute PH Diagram", key=f"button_{image_file}"):
            if not selected_indices:
                st.warning("Please select at least one contour.")
            else:
                with st.spinner("Computing..."):
                    selected_contours = [contours[i] for i in selected_indices]
                    data_cloud = np.concatenate([contour.squeeze() for contour in selected_contours])
                    res_pc = PersistenceDiagram.compute(data_cloud, maxdim=1)
                    
                    fig = plt.figure()
                    title = f'PH Diagram for contours {selected_indices}'
                    PersistenceDiagram.plot_persistence_diagram(res_pc['dgms'], title=title)
                    
                    # Store the computed figure in session state
                    st.session_state.ph_diagrams[image_file] = fig
                    st.rerun()
    else:
        st.info("No contours to compute PH diagram.")
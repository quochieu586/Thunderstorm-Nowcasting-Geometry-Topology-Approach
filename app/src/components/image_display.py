import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from app.src.utils import create_indexed_contour_image

def display_images(img_for_contours, contours):
    """
    Displays the source image and the indexed contour image.
    """
    if img_for_contours is not None:
        st.image(img_for_contours, caption="Source Image", width='stretch')
    
    if img_for_contours is not None:
        if contours:
            contour_fig = create_indexed_contour_image(img_for_contours.shape, contours)
            st.pyplot(contour_fig)
        else:
            fig, ax = plt.subplots()
            ax.imshow(np.ones(img_for_contours.shape, dtype=np.uint8) * 255)
            ax.text(img_for_contours.shape[1]/2, img_for_contours.shape[0]/2, "No contours found", ha='center', va='center')
            ax.axis('off')
            st.pyplot(fig)

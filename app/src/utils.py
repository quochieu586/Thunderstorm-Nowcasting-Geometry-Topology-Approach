import streamlit as st
import os
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import datetime

from utils.preprocessing.functions import extract_contour_by_dbz
from app.src.config import LEGEND_DIR

# --- Caching Functions for Performance ---
@st.cache_data
def load_image(image_path):
    """Loads an image from a given path and converts it to RGB."""
    source_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if source_img is None:
        return None
    return cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

@st.cache_data
def get_contours(image_path, dbz_threshold, contour_area_threshold):
    """Extracts contours from an image based on dBZ and area thresholds."""
    with open(os.path.join(LEGEND_DIR, 'color_dbz.json')) as f:
        list_color = json.load(f)
    sorted_color = sorted({tuple(color[1]): color[0] for color in list_color}.items(), key=lambda item: item[1])

    source_img = load_image(image_path)
    if source_img is None:
        return None, None

    _, contours, _ = extract_contour_by_dbz(source_img, thresholds=[dbz_threshold], sorted_color=sorted_color)
    if not contours:
        return source_img, []

    contours = contours[0]
    contours = sorted([polygon for polygon in contours if cv2.contourArea(polygon) > contour_area_threshold], key=lambda x: cv2.contourArea(x), reverse=True)
    
    return source_img, contours

def create_indexed_contour_image(image_shape, contours):
    """Creates an image with indexed contours and a legend."""
    fig, ax = plt.subplots()
    ax.imshow(np.ones(image_shape, dtype=np.uint8) * 255)  # White background

    # Define a color cycle for contours
    colors = plt.cm.get_cmap('tab10', len(contours))

    for i, contour in enumerate(contours):
        # Draw contour using matplotlib
        poly = plt.Polygon(contour.squeeze(), closed=True, fill=None, edgecolor=colors(i), label=f'Contour {i}')
        ax.add_patch(poly)
        
        # Add text label near the contour's center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            ax.text(cX, cY, str(i), fontsize=2 * np.log(cv2.contourArea(contour)), color='black', ha='center', va='center')

    ax.legend()
    ax.axis('off')
    plt.tight_layout()
    return fig

def get_datetime_from_filename(filename: str) -> str:
    """Extracts datetime from filename and formats it."""
    try:
        base_name = os.path.splitext(filename)[0]
        dt_obj = datetime.datetime.strptime(base_name, '%Y%m%d-%H%M%S')
        return dt_obj.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, IndexError):
        return "N/A"

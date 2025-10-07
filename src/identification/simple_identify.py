import numpy as np
import cv2
from typing import List, Tuple
import numpy as np

from src.preprocessing import _preprocess, _convert_to_dbz
from .base import BaseStormIdentifier

from src.preprocessing import sorted_color

class SimpleContourIdentifier(BaseStormIdentifier):
    """
        Detect storm objects solely based on the contiguous spatial areas of pixels exceeding specified dBZ thresholds. 
    """
    def identify_storm(self, img: np.ndarray, threshold: int, filter_area: int) -> list[np.ndarray]:
        """
            Draw the DBZ contour for the image.

            Args:
                img: source image.
                thresholds: dbz thresholds for drawing contours.
                sorted_color: a list of tuples where each tuple includes:
                    - a color represented as an RGB triplet (e.g., (R, G, B)).
                    - a corresponding dBZ value, sorted in increasing order of dBZ.
            
            Returns:
                List[np.ndarray]: A list of detected contours, each represented as an array of points.
        """
        img = _preprocess(img)
        dbz_map = _convert_to_dbz(img, sorted_color).astype(np.uint8)

        # Get the region
        region = (dbz_map >= threshold).astype(np.uint8)

        # Draw the contour
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted([polygon for polygon in contours if cv2.contourArea(polygon) >= filter_area], 
                        key=lambda x: cv2.contourArea(x), reverse=True)

        return contours
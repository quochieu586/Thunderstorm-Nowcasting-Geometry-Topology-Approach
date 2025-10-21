import numpy as np
import cv2
import streamlit as st

from src.cores.base import StormObject, StormsMap
from src.preprocessing import convert_polygons_to_contours

class ContourWindow:

    def render(self, original_image: np.ndarray, storms_map: StormsMap) -> None:
        """Display processing results"""
        st.subheader("Radar Scan with Storm Identification")
        
        blank_img = np.ones_like(original_image) * 255  # White background
        contours_image = self._draw_contours_img(blank_img, storms_map.storms)

        st.image(original_image, caption="Original Radar Scan", width='stretch')
        st.image(contours_image, caption="Identified Storm Contours", width='stretch')

    def _draw_contours_img(self, blank_img: np.ndarray, storms: list[StormObject]) -> np.ndarray:
        height, width = blank_img.shape[:2]

        # ==== DRAW GRID ====
        grid_size = 50  # pixel step
        grid_color = (220, 220, 220)  # light gray
        thickness = 1

        # Draw vertical lines
        for x in range(0, width, grid_size):
            cv2.line(blank_img, (x, 0), (x, height), grid_color, thickness)
            cv2.putText(blank_img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        # Draw horizontal lines
        for y in range(0, height, grid_size):
            cv2.line(blank_img, (0, y), (width, y), grid_color, thickness)
            cv2.putText(blank_img, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        # ==== DRAW AXES (Ox, Oy) ====
        origin = (0, height - 1)  # bottom-left as (0,0)
        axis_color = (0, 0, 255)
        cv2.line(blank_img, origin, (width, height - 1), axis_color, 2)  # Ox
        cv2.line(blank_img, origin, (0, 0), axis_color, 2)               # Oy

        # Axis labels
        cv2.putText(blank_img, "Ox", (width - 30, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1)
        cv2.putText(blank_img, "Oy", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1)

        # ==== DRAW STORMS ====
        contours = convert_polygons_to_contours([storm.contour for storm in storms])
        
        # Draw each storm's contour on the image
        for contour, storm in zip(contours, storms):
            cv2.drawContours(blank_img, [contour], -1, storm.contour_color, thickness=1)
            cv2.circle(img=blank_img, center=(int(storm.contour.centroid.x), int(storm.contour.centroid.y)), radius=2, color=storm.contour_color, thickness=-1)

            # Compute centroid history using the history movement vectors. Note that, the centroid of current storm is final point.
            history_centroids = [(storm.contour.centroid.x, storm.contour.centroid.y)]
            for movement_vector in storm.history_movements[::-1]:
                prev_centroid = (history_centroids[0][0] - movement_vector[0], history_centroids[0][1] - movement_vector[1])
                history_centroids = [prev_centroid] + history_centroids

            # Draw arrow indicating movement direction
            for point_start, point_end in zip(history_centroids[:-1], history_centroids[1:]):
                cv2.arrowedLine(blank_img, (int(point_start[0]), int(point_start[1])),
                                (int(point_end[0]), int(point_end[1])), (0,0,0), 1, tipLength=0.1)

        return blank_img
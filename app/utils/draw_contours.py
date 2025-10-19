import numpy as np
import cv2

from .history_tracking_storms import StormWithMovements
from src.preprocessing import convert_polygons_to_contours

def draw_contours(blank_img: np.ndarray, storms: list[StormWithMovements]) -> np.ndarray:
    contours = convert_polygons_to_contours([storm.contour for storm in storms])
    
    # Draw each storm's contour on the image
    for contour, storm in zip(contours, storms):
        cv2.drawContours(blank_img, [contour], -1, storm.contour_color, thickness=1)

        # Draw centroid history
        for point in storm.history_centroids:
            cv2.circle(blank_img, (int(point[0]), int(point[1])), 3, storm.contour_color, -1)

        # Draw arrow indicating movement direction
        for point_start, point_end in zip(storm.history_centroids[:-1], storm.history_centroids[1:]):
            cv2.arrowedLine(blank_img, (int(point_start[0]), int(point_start[1])),
                            (int(point_end[0]), int(point_end[1])), storm.contour_color, 1, tipLength=0.3)

    return blank_img
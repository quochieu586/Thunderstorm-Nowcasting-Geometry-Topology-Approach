import numpy as np
import cv2

from src.cores.base import StormObject
from src.preprocessing import convert_polygons_to_contours

def draw_contours(blank_img: np.ndarray, storms: list[StormObject]) -> np.ndarray:
    contours = convert_polygons_to_contours([storm.contour for storm in storms])
    
    # Draw each storm's contour on the image
    for contour, storm in zip(contours, storms):
        cv2.drawContours(blank_img, [contour], -1, storm.contour_color, thickness=1)
        cv2.circle(img=blank_img, center=(int(storm.contour.centroid.x), int(storm.contour.centroid.y)), radius=2, color=storm.contour_color, thickness=-1)

        # Compute centroid history using the history movement vectors. Note that, the centroid of current storm is final point.
        history_centroids = [(storm.contour.centroid.x, storm.contour.centroid.y)]
        for movement_vector in storm.history_movements:
            prev_centroid = (history_centroids[0][0] - movement_vector[0], history_centroids[0][1] - movement_vector[1])
            history_centroids = [prev_centroid] + history_centroids

        # Draw arrow indicating movement direction
        for point_start, point_end in zip(history_centroids[:-1], history_centroids[1:]):
            cv2.arrowedLine(blank_img, (int(point_start[0]), int(point_start[1])),
                            (int(point_end[0]), int(point_end[1])), (0,0,0), 1, tipLength=0.1)

    return blank_img
import numpy as np
import cv2

from shapely.geometry import Polygon
from src.cores.contours import StormObject

class StormWithMovements(StormObject):
    history_centroids: list[tuple[int, int]]
    contour_color: tuple[int, int, int]

    def __init__(self, polygon: Polygon, id: str) -> None:
        super().__init__(contour=polygon, id=id)
        self.history_centroids = [polygon.centroid.coords[0]]
        self.contour_color = tuple(np.random.randint(0, 255, size=3).tolist())

    def update_history(self, prev_storms: 'StormWithMovements') -> None:
        """
        Update the history of centroids with the previous storm's centroids history
        """
        self.history_centroids.extend(prev_storms.history_centroids)
        self.contour_color = prev_storms.contour_color


import numpy as np
from shapely.geometry import Polygon

from src.cores.base import StormObject

class CentroidStorm(StormObject):
    centroid: np.ndarray
    estimated_velocity: tuple[float, float] = None  # estimated movement for the current storm in the next frame

    def __init__(self, polygon: Polygon, centroid: tuple[float, float], id: str=""):
        super().__init__(polygon, id)
        self.centroid = np.array(centroid)
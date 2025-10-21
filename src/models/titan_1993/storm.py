import numpy as np
from shapely import Polygon
from typing import Optional

from src.cores.base import StormObject

class CentroidStorm(StormObject):
    centroid: np.ndarray

    def __init__(self, polygon: Polygon, centroid: tuple[float, float], id: str=""):
        super().__init__(contour=polygon, id=id)
        self.centroid = np.array(centroid)
  
    def tracking_history(self, merged_move: list[tuple[float, float]], contour_color_updated: Optional[np.ndarray]=None):
        """
        Feed directly the merged movement list into the history movements.
        """
        self.history_movements = merged_move
        if contour_color_updated is not None:
            self.contour_color = contour_color_updated
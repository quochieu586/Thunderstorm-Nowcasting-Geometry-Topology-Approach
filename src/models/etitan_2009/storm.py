import numpy as np
import cv2
from typing import Optional
from shapely.geometry import Polygon
from skimage.measure import block_reduce

from src.cores.base import StormObject
from src.preprocessing import convert_polygons_to_contours

class CentroidStorm(StormObject):
    centroid: np.ndarray
    estimated_movement: tuple[float, float] = None

    def __init__(self, polygon: Polygon, centroid: tuple[float, float], img_shape: tuple[int, int], id: str=""):
        super().__init__(contour=polygon, id=id)
        self.centroid = np.array(centroid)
        self.img_shape = img_shape

    def retrieve_movement(self, **kargs):
        """
        Retrieve movement of the current storm by combining movement from
        """
        if self.estimated_movement is not None:
            return self.estimated_movement

        # retrieve parameters of estimatied-motion map.
        grid_y = kargs.get("grid_y", np.array([]))
        grid_x = kargs.get("grid_x", np.array([]))
        vy = kargs.get("vy", np.array([]))
        vx = kargs.get("vx", np.array([]))

        if len(grid_y) == 0:
            return np.array([0,0])

        block_size = int(grid_y[0] * 2)

        contours = convert_polygons_to_contours([self.contour])
        mask = np.zeros(self.img_shape, dtype=np.uint8)
        cv2.fillPoly(mask, contours, color=1)

        crop_mask = mask[0:block_size * len(grid_y), 0:block_size * len(grid_x)]

        block_mask = block_reduce(crop_mask, block_size=(block_size,block_size), func=np.sum)
        total = np.sum(block_mask) + 1e-8
        dy = np.sum(vy * block_mask) / total
        dx = np.sum(vx * block_mask) / total

        self.estimated_movement = (dy, dx)

        return dy, dx
    
    def tracking_history(self, merged_move: list[tuple[float, float]], contour_color_updated: Optional[np.ndarray]=None):
        """
        Feed directly the merged movement list into the history movements.
        """
        self.history_movements = merged_move
        if contour_color_updated is not None:
            self.contour_color = contour_color_updated
import numpy as np
import cv2
from shapely.geometry import Polygon
from skimage.measure import block_reduce

from src.cores.base import StormObject
from src.preprocessing import convert_polygons_to_contours

class CentroidStorm(StormObject):
    """
    Storm object represented by its centroid for tracking purposes. This storm object is customized to work in ETitan model.

    Attributes:
        centroid (np.ndarray): The centroid coordinates of the storm.
        history_movements (list[tuple[float, float]]): List of historical vector movements of the storm.
        estimated_movement (tuple[float, float], optional): The estimated movement vector of the storm to predict next position
    """
    centroid: np.ndarray
    history_movements: list[tuple[float, float]]
    estimated_movement: tuple[float, float] = None

    def __init__(self, polygon: Polygon, centroid: tuple[float, float], id: str="", img_shape: tuple[int, int]=(512, 512)):
        super().__init__(contour=polygon, id=id)
        self.centroid = np.array(centroid)
        self.img_shape = img_shape
        self.history_movements = []

    def retrieve_movement(self, grid_x: np.ndarray = np.array([]), grid_y: np.ndarray = np.array([]), vx: np.ndarray = np.array([]), vy: np.ndarray = np.array([])) -> tuple[float, float]:
        """
        
        """
        if self.estimated_movement is not None:
            return self.estimated_movement

        if len(grid_y) == 0:
            return (0, 0)

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
    
    def track_history(self, previous_storm: "CentroidStorm"):
        # Transfer historical movements from previous storm
        self.history_movements = previous_storm.history_movements.copy()
        self.history_movements.append((self.estimated_movement if self.estimated_movement is not None else (0, 0)))
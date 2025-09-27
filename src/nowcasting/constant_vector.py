import numpy as np
import cv2

from .base import BaseStormMotion
from src.cores.contours import StormObject

class ConstantVectorMotion(BaseStormMotion):
    """
        Simplest motion estimation: assume motion following single vector between consecutive images and no change on shape. 
    """
    storms: list[StormObject]
    x_vector: float
    y_vector: float

    def __init__(self):
        self.storms = []
        self.movement_vectors = []
        self.x_vector = 0.0
        self.y_vector = 0.0

    def estimate_motion(self, storm_object: StormObject) -> StormObject:
        storm_object.contour[:, 0] += self.x_vector
        storm_object.contour[:, 1] += self.y_vector

        return storm_object

    def __add__(self, other: 'ConstantVectorMotion') -> 'ConstantVectorMotion':
        combined = ConstantVectorMotion()
        combined.x_vector = self.x_vector + other.x_vector
        combined.y_vector = self.y_vector + other.y_vector
        return combined
    
    def _compute_centroid(self, contour: np.ndarray) -> tuple[float, float]:
        """
            Compute the centroid of a contour.
            Args:
                contour (np.ndarray): The contour points.
            Returns:
                tuple: The (x, y) coordinates of the centroid.
        """
        M = cv2.moments(contour)

        if M["m00"] == 0:
            return (0.0, 0.0)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return (cx, cy)

    def update_motion(self, new_storm: StormObject) -> None:
        self.storms.append(new_storm)

        if len(self.storms) < 2:
            return
        else:
            new_move_vector = np.array(self._compute_centroid(self.storms[-1].contour)) - np.array(self._compute_centroid(self.storms[-2].contour))
            self.movement_vectors.append(new_move_vector)
            self.x_vector = np.mean([vec[0] for vec in self.movement_vectors])
            self.y_vector = np.mean([vec[1] for vec in self.movement_vectors])

            return


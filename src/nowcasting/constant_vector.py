import numpy as np
import cv2
from shapely.affinity import translate

from .base import BaseStormMotion
from src.cores.base import StormObject

class ConstantVectorMotion(BaseStormMotion):
    """
        Simplest motion estimation: assume motion following single vector between consecutive images and no change on shape.
        Use exponential decay to reduce the effect of previous vectors.
    """
    movement_vectors: list[tuple[float, float]]
    x_vector: float
    y_vector: float

    def __init__(self, exponential_decay: float = 0.7) -> None:
        self.movement_vectors = []
        self.x_vector = 0.0
        self.y_vector = 0.0
        self.exponential_decay = exponential_decay

    def estimate_motion(self, storm_object: StormObject) -> StormObject:
        new_object = storm_object.copy()
        new_object.contour = translate(new_object.contour, xoff=self.x_vector, yoff=self.y_vector)

        return new_object

    def __add__(self, other: 'ConstantVectorMotion') -> 'ConstantVectorMotion':
        combined = ConstantVectorMotion()
        combined.x_vector = self.x_vector + other.x_vector
        combined.y_vector = self.y_vector + other.y_vector
        return combined

    def update_motion(self, new_translate: tuple[float, float]) -> None:
        self.movement_vectors.append(new_translate)
        self.x_vector = self.exponential_decay * self.x_vector + (1 - self.exponential_decay) * new_translate[0]
        self.y_vector = self.exponential_decay * self.y_vector + (1 - self.exponential_decay) * new_translate[1]

        return None


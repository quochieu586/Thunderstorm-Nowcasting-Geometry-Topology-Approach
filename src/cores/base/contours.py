from dataclasses import dataclass
import numpy as np
from datetime import datetime
from typing import Optional, Union

from .base_object import BaseObject
from shapely.geometry import Polygon
from shapely.affinity import translate
from src.preprocessing import convert_contours_to_polygons

@dataclass
class StormObject(BaseObject):
    contour: Polygon    # List of points represent contours
    contour_color: tuple[int, int, int]
    id: str    # Unique ID of this object for tracking over time
    original_id: str
    history_movements: list[tuple[float, float]]
    centroid: tuple[float, float] = None

    def __init__(
            self, contour: Union[Polygon, np.ndarray], 
            history_movements: Optional[list[tuple[float, float]]] = [],
            centroid: tuple[float, float] = None, id: str = ""
        ):
        if type(contour) is np.ndarray:
            contour = convert_contours_to_polygons([contour])[0]

        self.contour = contour
        self.centroid = centroid
        self.id = id
        self.original_id = id
        self.history_movements = history_movements
        self.contour_color = tuple(np.random.randint(0, 255, size=3).tolist())

    def copy(self) -> 'StormObject':
        return StormObject(
            contour=self.contour,
            centroid=self.centroid,
            id=self.id,
            history_movements=self.history_movements.copy()
        )

    def clear_history(self):
        self.history_movements = []

    def track_history(self, previous_storm: "StormObject", optimal_movement: tuple[float, float] = (0, 0)):
        # Transfer historical movements from previous storm
        self.history_movements = previous_storm.history_movements.copy()
        self.contour_color = previous_storm.contour_color
        
        self.history_movements.append(optimal_movement)
    
    def make_move(self, movement: tuple[float, float]) -> None:
        """
        Move the storm by the given movement vector.

        Args:
            movement (tuple[float, float]): Movement vector (dx, dy).
        """
        dy, dx = movement
        self.contour = translate(self.contour, xoff=dx, yoff=dy)
        if self.centroid is not None:
            cy, cx = self.centroid
            self.centroid = (cy + dy, cx + dx)
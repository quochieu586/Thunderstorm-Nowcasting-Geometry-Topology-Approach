from dataclasses import dataclass
import numpy as np
from datetime import datetime
from typing import Optional, Union

from .base_object import BaseObject
from shapely.geometry import Polygon
from src.preprocessing import convert_contours_to_polygons

THRESHOLD_DBZ = 30

@dataclass
class StormObject(BaseObject):
    contour: Polygon    # List of points represent contours
    history_movements: list[tuple[float, float]]
    id: str = ""        # Unique ID of this object for tracking over time

    def __init__(self, contour: Union[Polygon, np.ndarray], id: str = ""):
        if type(contour) is np.ndarray:
            contour = convert_contours_to_polygons([contour])[0]

        self.contour = contour
        self.id = id
        self.history_movements = []

    def copy(self) -> 'StormObject':
        return StormObject(
            contour=self.contour,
            id=self.id,
            history_movements=self.history_movements.copy()
        )

    def clear_history(self):
        self.history_movements = []

    def track_history(self, previous_storm: "StormObject", optimal_movement: tuple[float, float] = (0, 0)):
        # Transfer historical movements from previous storm
        self.history_movements = previous_storm.history_movements.copy()
        self.history_movements.append(optimal_movement)

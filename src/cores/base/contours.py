from dataclasses import dataclass
import random
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
    contour_color: tuple[int, int, int]
    id: str    # Unique ID of this object for tracking over time
    original_id: str

    def __init__(self, contour: Union[Polygon, np.ndarray], id: str = ""):
        if type(contour) is np.ndarray:
            contour = convert_contours_to_polygons([contour])[0]

        self.contour = contour
        self.id = id
        self.original_id = id
        self.history_movements = []
        self.contour_color = tuple(np.random.randint(0, 255, size=3).tolist())

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
        self.contour_color = previous_storm.contour_color
        
        self.history_movements.append(optimal_movement)

    def plot_on(self, ax, color=None, label=None):
        """
        Plot the storm contour and its sampled particles on the given axes,
        using the same color for both.

        Args:
            ax: matplotlib Axes object to draw on.
            color: Optional color for the contour and particles (random if None).
            label: Optional label for the storm.
        """

        # Pick a random color if none provided
        if color is None:
            color = (random.random(), random.random(), random.random())

        # --- Draw the storm contour (polygon boundary) ---
        x, y = self.contour.exterior.xy
        ax.plot(x, y, color=color, linewidth=2, label=label or self.id)

        # --- Draw sampled particle centers (same color as contour) ---
        if hasattr(self, "shape_vectors") and len(self.shape_vectors) > 0:
            coords = np.array([sv.coord for sv in self.shape_vectors])
            ax.scatter(coords[:, 0], coords[:, 1], s=10, color=color, marker="o")

        ax.set_aspect("equal")
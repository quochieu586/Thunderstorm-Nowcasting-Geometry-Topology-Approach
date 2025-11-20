from dataclasses import dataclass
import random
import numpy as np
from datetime import datetime
from typing import Optional

from ..base import BaseObject
from shapely.geometry import Polygon

THRESHOLD_DBZ = 30

@dataclass
class StormObject(BaseObject):
    contour: Polygon    # List of points represent contours
    id: str = ""        # Unique ID of this object for tracking over time

    def copy(self) -> 'StormObject':
        return StormObject(
            contour=self.contour,
            id=self.id,
        )
    
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


from dataclasses import dataclass
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


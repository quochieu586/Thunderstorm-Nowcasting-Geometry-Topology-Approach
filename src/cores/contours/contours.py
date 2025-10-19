from dataclasses import dataclass
import numpy as np
from datetime import datetime
from typing import Optional

from shapely.geometry import Polygon

THRESHOLD_DBZ = 30

@dataclass
class StormObject:
    contour: Polygon                                         # List of points represent contours
    max_dbz_centroid: int = THRESHOLD_DBZ                       # Maximum dBZ value of this object
    id: str = ""                                            # Unique ID of this object for tracking over time
    time_frame: Optional[datetime] = None

    def copy(self) -> 'StormObject':
        return StormObject(
            contour=self.contour,
            max_dbz_centroid=self.max_dbz_centroid,
            id=self.id,
            time_frame=self.time_frame
        )

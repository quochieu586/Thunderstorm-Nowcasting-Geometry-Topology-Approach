from dataclasses import dataclass
import numpy as np
from datetime import datetime
from typing import Optional

from src.nowcasting import BaseStormMotion, ConstantVectorMotion

THRESHOLD_DBZ = 30

@dataclass
class StormObject:
    contour: np.ndarray                                         # List of points represent contours
    max_dbz_centroid: int = THRESHOLD_DBZ                       # Maximum dBZ value of this object
    id: int = -1
    time_frame: Optional[datetime] = None
    local_motion: BaseStormMotion = ConstantVectorMotion()      # Local motion of this object
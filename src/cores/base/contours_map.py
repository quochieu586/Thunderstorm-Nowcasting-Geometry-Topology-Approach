from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from .contours import StormObject

import numpy as np

THRESHOLD_DBZ = 30

@dataclass
class StormsMap:
    storms: list[StormObject]
    time_frame: datetime
    dbz_map: Optional[np.ndarray] = None
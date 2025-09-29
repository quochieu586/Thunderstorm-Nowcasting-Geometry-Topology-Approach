from dataclasses import dataclass
from datetime import datetime

from .contours import StormObject

THRESHOLD_DBZ = 30

@dataclass
class StormsMap:
    storms: list[StormObject]
    time_frame: datetime
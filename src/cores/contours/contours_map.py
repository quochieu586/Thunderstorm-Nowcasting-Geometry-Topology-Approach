from dataclasses import dataclass
from datetime import datetime

from src.nowcasting import BaseStormMotion
from .contours import StormObject

THRESHOLD_DBZ = 30

@dataclass
class StormsMap:
    global_motion: BaseStormMotion
    storms: list[StormObject]
    time_frame: datetime
    map_size: tuple[int, int]
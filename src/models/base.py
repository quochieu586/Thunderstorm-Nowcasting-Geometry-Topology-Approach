from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
from copy import deepcopy

from src.cores.base import StormsMap
from src.preprocessing import convert_contours_to_polygons

class BasePrecipitationModeling(ABC):
    """
    Simple precipitation modeling using contour-based storm identification.

    Attributes:
        identifier (SimpleContourIdentifier): The storm identifier used for identifying storms in radar images.
    """
    storms_maps: list[StormsMap]

    def __init__(self):
        pass

    @abstractmethod
    def identify_storms(self, dbz_img: np.ndarray, time_frame: datetime, map_id: str, threshold: float, filter_area: float) -> StormsMap:
        pass

    @abstractmethod
    def processing_map(self, current_map: StormsMap):
        """
        Add new map into the tracking system and update the previous storms accordingly.
        """
        pass

    def copy(self):
        """
        Create a copy of the current model instance. This copy is mutable and independent of the original instance.
        """
        new_instance = deepcopy(self)
        new_instance.storms_maps = []

        return new_instance

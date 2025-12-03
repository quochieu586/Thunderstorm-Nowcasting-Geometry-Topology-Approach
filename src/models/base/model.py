from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
from copy import deepcopy

from .tracker import TrackingHistory
from src.cores.base import StormsMap

class BasePrecipitationModel(ABC):
    """
    Simple precipitation modeling using contour-based storm identification.

    Attributes:
        identifier (SimpleContourIdentifier): The storm identifier used for identifying storms in radar images.
    """
    storms_maps: list[StormsMap]
    tracking_history: TrackingHistory = TrackingHistory()

    @abstractmethod
    def identify_storms(self, dbz_img: np.ndarray, time_frame: datetime, map_id: str, threshold: float, filter_area: float) -> StormsMap:
        pass

    @abstractmethod
    def processing_map(self, current_map: StormsMap) -> int:
        """
        Process the current StormsMap and update the model's internal state: append new storms map and update movement history for all tracked storms.

        Returns:
            num_of_matched_storms (int): Number of storms that can be found history in previous maps.
        """
        pass

    @abstractmethod
    def forecast(self, lead_time: float) -> StormsMap:
        """
        Forecast the storms map for a given lead time based on the current model state.

        Args:
            lead_time (float): The lead time in hours for which to forecast the storms map.
        Returns:
            StormsMap: The forecasted storms map at the specified lead time.
        """
        pass

    def copy(self):
        """
        Create a copy of the current model instance. This copy is mutable and independent of the original instance.
        """
        new_instance = deepcopy(self)
        new_instance.storms_maps = []

        return new_instance


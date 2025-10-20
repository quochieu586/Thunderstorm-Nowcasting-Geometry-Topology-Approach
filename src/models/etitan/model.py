import numpy as np
from datetime import datetime

from src.models.base import BasePrecipitationModeling
from src.identification import MorphContourIdentifier

from .matcher import Matcher
from .tracker import Tracker, TrackingHistorys
from .storm_map import etitan_identify_storms_map, DbzStormsMap
from .storm import CentroidStorm

class ETitanModel(BasePrecipitationModeling):
    """
    ETitan model implementation for thunderstorm nowcasting.
    """
    identifier: MorphContourIdentifier
    dbz_maps: list[DbzStormsMap]
    tracker: Tracker

    def __init__(self, identifier: MorphContourIdentifier):
        self.identifier = identifier
        self.dbz_maps = []
        self.tracker = Tracker(dynamic_max_velocity=self._dynamic_max_velocity)

    def _dynamic_max_velocity(self, area: float) -> float:
        """
        Dynamic constraint for maximum velocity based on storm area. The unit of velocity is pixel/hr.
        """
        if area < 300:
            return 500
        elif area < 500:
            return 750
        else:
            return 1000

    def identify_storms(self, dbz_img: np.ndarray, timestamp: datetime) -> DbzStormsMap:
        """
        Identify storms in the given DBZ image at the specified timestamp.

        Args:
            dbz_img (np.ndarray): The DBZ image.
            timestamp (datetime): The timestamp of the image.

        Returns:
            DbzStormsMap: The identified storms map.
        """
        storm_map = etitan_identify_storms_map(dbz_img, timestamp)
        return storm_map
    
    def processing_map(self, storm_map: DbzStormsMap):
        self.dbz_maps.append(storm_map)

        if len(self.dbz_maps) > 2:
            # Start matching to trace history
            previous_map = self.dbz_maps[-2]
            current_map = self.dbz_maps[-1]

            matching_assignments = self.tracker.estimate_matching(previous_map, current_map)

            for prev_idx, curr_idx in matching_assignments:
                previous_storm = previous_map.storms[prev_idx]
                current_storm = current_map.storms[curr_idx]
                current_storm.track_history(previous_storm)
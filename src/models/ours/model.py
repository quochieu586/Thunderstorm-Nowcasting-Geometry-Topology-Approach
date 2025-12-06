import numpy as np
from datetime import datetime, timedelta

from src.cores.base import StormsMap
from src.identification import BaseStormIdentifier, HypothesisIdentifier
from src.preprocessing import convert_contours_to_polygons
from src.models.base.model import BasePrecipitationModel
from src.models.base.tracker import TrackingHistory, UpdateType
from .matcher import StormMatcher
from .storm import DbzStormsMap, ShapeVectorStorm

from .default import DEFAULT_MAX_VELOCITY, DEFAULT_WEIGHTS, DEFAULT_COARSE_MATCHING_THRESHOLD, DEFAULT_FINE_MATCHING_THRESHOLD

class OursPrecipitationModel(BasePrecipitationModel):
    """
    Simple precipitation modeling using contour-based storm identification.

    Attributes:
        identifier (SimpleContourIdentifier): The storm identifier used for identifying storms in radar images.
    """
    identifier: HypothesisIdentifier
    matcher: StormMatcher
    tracker: TrackingHistory
    storms_maps: list[StormsMap]

    def __init__(self, identifier: HypothesisIdentifier, max_velocity: float = DEFAULT_MAX_VELOCITY, weights: tuple[float, float] = DEFAULT_WEIGHTS):
        self.identifier = identifier
        self.storms_maps = []
        self.matcher = StormMatcher(max_velocity=max_velocity, weights=weights)
        self.tracker = None

    def identify_storms(self, dbz_img: np.ndarray, time_frame: datetime, map_id: str, threshold: int, filter_area: float) -> StormsMap:
        contours = self.identifier.identify_storm(dbz_img, threshold=threshold, filter_area=filter_area)
        polygons = convert_contours_to_polygons(contours)
        polygons = sorted(polygons, key=lambda x: x.area, reverse=True)

        # Construct storms map
        storms = [ShapeVectorStorm(
                    polygon=polygon, 
                    id=f"{map_id}_storm_{idx}",
                    dbz_map=dbz_img
                ) for idx, polygon in enumerate(polygons)]
        
        return DbzStormsMap(storms, time_frame=time_frame, dbz_map=dbz_img)

    def processing_map(self, curr_storms_map: StormsMap, coarse_threshold: float = DEFAULT_COARSE_MATCHING_THRESHOLD, fine_threshold: float = DEFAULT_FINE_MATCHING_THRESHOLD) -> int:
        if len(self.storms_maps) == 0:
            self.tracker = TrackingHistory(curr_storms_map)
            update_list = []
        else:
            prev_storms_map = self.storms_maps[-1]
            if curr_storms_map.time_frame <= prev_storms_map.time_frame:
                raise ValueError("Current storms map time frame must be later than the previous one.")
            
            update_list = self.matcher.match_storms(
                storms_map_lst_1=prev_storms_map,
                storms_map_lst_2=curr_storms_map,
                coarse_threshold=coarse_threshold,
                fine_threshold=fine_threshold
            )

            for info in update_list:
                if info.update_type == UpdateType.NEW:
                    self.tracker.add_new_track(
                        new_storm=curr_storms_map.storms[info.curr_storm_order],
                        time_frame=curr_storms_map.time_frame
                    )
                else:
                    self.tracker.update_track(
                        prev_storm=prev_storms_map.storms[info.prev_storm_order],
                        curr_storm=curr_storms_map.storms[info.curr_storm_order],
                        update_type=info.update_type,
                        time_frame=curr_storms_map.time_frame,
                        velocity=info.velocity
                    )

        self.storms_maps.append(curr_storms_map)
        right_matches = list(set([info.curr_storm_order for info in update_list]))
        return len(right_matches)
    
    def forecast(self, lead_time: float) -> StormsMap:
        """
        Predict future storms up to lead_time based on the current storm map.

        Args:
            lead_time (float): The lead time in second for prediction.
        """
        if self.tracker is None:
            raise ValueError("No storm history available for prediction. Please process at least one storm map.")
        
        lead_time_hours = lead_time / 3600.0

        return StormsMap([
            self.tracker.forecast(storm.id, lead_time_hours) for storm in self.storms_maps[-1].storms
        ], time_frame=self.storms_maps[-1].time_frame + timedelta(hours=lead_time_hours))
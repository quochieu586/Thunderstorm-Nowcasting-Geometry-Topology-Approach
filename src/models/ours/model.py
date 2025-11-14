import numpy as np
from datetime import datetime, timedelta

from src.cores.base import StormsMap
from src.identification import BaseStormIdentifier, HypothesisIdentifier
from src.preprocessing import convert_contours_to_polygons

from ..base import BasePrecipitationModel
from .storm import ShapeVectorStorm
from .matcher import StormMatcher, MAX_VELOCITY, MATCHING_THRESHOLD
from .tracker import TrackingHistory

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

    def __init__(self, identifier: HypothesisIdentifier, max_velocity: float = MAX_VELOCITY, matching_threshold: float = MATCHING_THRESHOLD):
        self.identifier = identifier
        self.storms_maps = []
        self.matcher = StormMatcher(max_velocity=max_velocity, matching_threshold=matching_threshold)
        self.tracker = None

    def identify_storms(self, dbz_img: np.ndarray, time_frame: datetime, map_id: str, threshold: int, filter_area: float) -> StormsMap:
        contours = self.identifier.identify_storm(dbz_img, threshold=threshold, filter_area=filter_area)
        polygons = convert_contours_to_polygons(contours)
        polygons = sorted(polygons, key=lambda x: x.area, reverse=True)

        # Construct storms map
        storms = [ShapeVectorStorm(
                    polygon=polygon, 
                    id=f"{map_id}_storm_{idx}",
                    global_contours=contours,
                    img_shape=dbz_img.shape[:2]
                ) for idx, polygon in enumerate(polygons)]
        
        return StormsMap(storms, time_frame=time_frame)

    def processing_map(self, curr_storms_map: StormsMap) -> int:
        if len(self.storms_maps) == 0:
            self.tracker = TrackingHistory(curr_storms_map)
            assignments = []
        else:
            prev_storms_map = self.storms_maps[-1]
            dt = (curr_storms_map.time_frame - prev_storms_map.time_frame).seconds / 3600   # scaled to hour

            # match using Hungarian algorithm
            ## the match result already includes split & merge
            assignments, scores, displacements = self.matcher.match_storms(prev_storms_map, curr_storms_map)
            mapping_curr = {curr_idx: [] for curr_idx in range(len(curr_storms_map.storms))}

            for (prev_idx, curr_idx), score, displacement in zip(assignments, scores, displacements):
                mapping_curr[int(curr_idx)] = [(int(prev_idx), score, displacement)]

            self.tracker.update(mapping_curr, prev_storms_map, curr_storms_map)

            # Update history movements to track history movement
            for storm in curr_storms_map.storms:
                storm_controller = self.tracker._get_track(storm.id)[0]
                storm.contour_color = storm_controller["storm_lst"][-1].contour_color if len(storm_controller) >= 1 else storm.contour_color
                storm.history_movements = [mv * dt for mv in storm_controller["movement"]]
            self.storms_maps.append(curr_storms_map)

        self.storms_maps.append(curr_storms_map)

        right_matches = list(set([curr for _, curr in assignments]))
        return len(right_matches)
    
    def prediction(self, lead_time: float) -> StormsMap:
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
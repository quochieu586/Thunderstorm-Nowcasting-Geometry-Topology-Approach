import numpy as np
from datetime import datetime, timedelta

from src.cores.base import StormsMap
from src.identification import BaseStormIdentifier, HypothesisIdentifier
from src.preprocessing import convert_contours_to_polygons

from ..base import BasePrecipitationModel
from .storm import ShapeVectorStorm
from .matcher import StormMatcher, MAX_VELOCITY, MATCHING_THRESHOLD
from .tracker import TrackingHistory

class NewPrecipitationModel(BasePrecipitationModel):
    """
    Simple precipitation modeling using contour-based storm identification.

    Attributes:
        identifier (SimpleContourIdentifier): The storm identifier used for identifying storms in radar images.
    """
    identifier: HypothesisIdentifier
    matcher: StormMatcher
    tracker: TrackingHistory
    storms_maps: list[StormsMap]

    def __init__(self, identifier: HypothesisIdentifier, max_velocity: float = MAX_VELOCITY, coarse_threshold: float = MATCHING_THRESHOLD, fine_threshold: float = MATCHING_THRESHOLD):
        self.identifier = identifier
        self.storms_maps = []
        self.matcher = StormMatcher(max_velocity=max_velocity)
        self.tracker = None

        self.coarse_threshold=coarse_threshold
        self.fine_threshold=fine_threshold

    def identify_storms(self, dbz_img: np.ndarray, time_frame: datetime, map_id: str, threshold: int, filter_area: float) -> StormsMap:
        contours = self.identifier.identify_storm(dbz_img, threshold=threshold, filter_area=filter_area)
        polygons = convert_contours_to_polygons(contours)
        polygons = sorted(polygons, key=lambda x: x.area, reverse=True)

        # Construct storms map
        storms = [ShapeVectorStorm(
                    polygon=polygon, 
                    id=f"{map_id}_storm_{idx}",
                    dbz_map=dbz_img,
                    img_shape=dbz_img.shape[:2]
                ) for idx, polygon in enumerate(polygons)]
        
        return StormsMap(storms, time_frame=time_frame)

    def processing_map(self, curr_storms_map: StormsMap) -> int:
        if len(self.storms_maps) == 0:
            self.tracker = TrackingHistory(curr_storms_map)
            matched = []
        else:
            prev_storms_map = self.storms_maps[-1]
            dt = (curr_storms_map.time_frame - prev_storms_map.time_frame).seconds / 3600   # scaled to hour

            # 1. Match storms between frames using Hungarian algorithm
            matched = self.matcher.match_storms(prev_storms_map, curr_storms_map, coarse_threshold=self.coarse_threshold, fine_threshold=self.fine_threshold)

            # 2. Build mapping from current → list of matched previous storms
            mapping_curr = {i: [] for i in range(len(curr_storms_map.storms))}

            for pair in matched:
                mapping_curr[pair.curr_storm_order].append((
                    pair.prev_storm_order,
                    pair.curr_score,
                    pair.prev_score,
                    pair.derive_motion_vector(dt),
                ))

            # 3. Resolve merge/split cases
            inherited = {}  # curr_idx → parent info
            parent_children = {i: [] for i in range(len(prev_storms_map.storms))}

            # 3a. Assign each current storm its best previous parent
            for curr_idx, matches in mapping_curr.items():
                if not matches:
                    continue

                # select previous storm with highest curr_score
                parent_prev_idx, _, prev_score, _ = max(matches, key=lambda x: x[1])

                inherited[curr_idx] = {'parent_id': parent_prev_idx, 'virtual': False}
                parent_children[parent_prev_idx].append((curr_idx, prev_score))

            # 3b. Detect splits: a previous storm inherited by >1 current storms
            for _, children in parent_children.items():
                if len(children) <= 1:
                    continue

                # sort by parent_score descending
                children_sorted = sorted(children, key=lambda x: x[1], reverse=True)

                # first keeps real ID, others become virtual (split)
                for curr_idx, _ in children_sorted[1:]:
                    inherited[curr_idx]['virtual'] = True

            # 4. Update tracker with resolved matching
            self.tracker.update(mapping_curr, inherited, prev_storms_map, curr_storms_map)

            # Update history movements to track history movement
            for storm in curr_storms_map.storms:
                storm_controller = self.tracker._get_track(storm.id)
                storm.contour_color = storm_controller.records[-1].storm.contour_color if len(storm_controller.records) >= 1 else storm.contour_color
                storm.history_movements = [mv * dt for mv in storm_controller.movements]
            self.storms_maps.append(curr_storms_map)

        self.storms_maps.append(curr_storms_map)

        right_matches = list(set([m.curr_storm_order for m in matched]))
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
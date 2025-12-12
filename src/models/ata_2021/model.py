import numpy as np
from datetime import datetime
import cv2

from ..base.base import BasePrecipitationModel
from src.identification import MorphContourIdentifier
from src.preprocessing import convert_contours_to_polygons, convert_polygons_to_contours

from .storm import CentroidStorm
from .storm_map import DbzStormsMap
from .tracker import TrackingHistory, Correspondence
from .matcher import Matcher

class AdaptiveTrackingPrecipitationModel(BasePrecipitationModel):
    """
    Precipitation modeling using adaptive contour-based storm identification and tracking. This is from Adaptive Tracker algorithm.

    Attributes:
        identifier (SimpleContourIdentifier): The storm identifier used for identifying storms in radar images.
    """
    identifier: MorphContourIdentifier
    matcher: Matcher
    tracker: TrackingHistory
    storms_maps: list[DbzStormsMap]

    def __init__(self, identifier: MorphContourIdentifier):
        self.identifier = identifier
        self.matcher = Matcher()
        self.tracker = None
        self.storms_maps = []

    def identify_storms(self, dbz_img: np.ndarray, time_frame: datetime, map_id: str, threshold: int, filter_area: float) -> DbzStormsMap:
        contours = self.identifier.identify_storm(dbz_img, threshold=threshold, filter_area=filter_area)
        polygons = convert_contours_to_polygons(contours)
        polygons = sorted(polygons, key=lambda x: x.area, reverse=True)

        # Keep list of storms
        storms = []

        for idx, polygon in enumerate(polygons):
            contour = convert_polygons_to_contours([polygon])[0]

            # Create the mask of current storm
            mask = np.zeros_like(dbz_img, dtype=np.uint8)
            cv2.fillPoly(mask, contour, color=1)

            # Extract DBZ values inside mask

            # Compute the geometric centroid instead of weighted centroid
            centroid = polygon.centroid.coords[0]

            storms.append(CentroidStorm(
                    polygon, centroid=centroid, id=f"{map_id}_storm_{idx}"
                ))
            
        return DbzStormsMap(storms, time_frame=time_frame, dbz_map=dbz_img)
    
    def processing_map(self, curr_storms_map: DbzStormsMap) -> int:
        if len(self.storms_maps) == 0:
            self.tracker = TrackingHistory(curr_storms_map)
            assignments = []
        else:
            prev_storms_map = self.storms_maps[-1]
            dt = (curr_storms_map.time_frame - prev_storms_map.time_frame).total_seconds() / 3600.0  # in hours

            # get historical velocities
            history_velocities = self.tracker.get_history_velocities(prev_storms_map.storms)
            # print(f"History velocities retrieved: {history_velocities}")

            # match storms
            assignments, disparity_matrix = self.matcher.match_storms(
                prev_storms_map, curr_storms_map, history_velocities
            )

            # resolve merge & split
            prev_correspondence_mapping = {}
            curr_correspondence_mapping = {}
            correspondence_lst: list[Correspondence] = []

            for prev_idx, curr_idx in assignments:
                correspondence_lst.append(Correspondence(
                    prev_indices=[prev_idx],
                    curr_indices=[curr_idx]
                ))
                prev_correspondence_mapping[prev_idx] = len(correspondence_lst) - 1
                curr_correspondence_mapping[curr_idx] = len(correspondence_lst) - 1
            
            ## resolve merge
            unassigned_prev = [i for i in range(len(prev_storms_map.storms)) if i not in prev_correspondence_mapping]
            for prev_idx in unassigned_prev:
                prev_pol = prev_storms_map.storms[prev_idx].contour
                curr_matched_idx = -1
                best_intersection_area = 0

                for curr_idx, curr_storm in enumerate(curr_storms_map.storms):
                    curr_pol = curr_storm.contour
                    if prev_pol.intersection(curr_pol).area > best_intersection_area:
                        best_intersection_area = prev_pol.intersection(curr_pol).area
                        curr_matched_idx = curr_idx
                
                # case: found the best match
                if best_intersection_area > 0:
                    if curr_matched_idx in curr_correspondence_mapping:
                        corr_id = curr_correspondence_mapping[curr_matched_idx]
                        correspondence_lst[corr_id].prev_indices.append(prev_idx)
                        prev_correspondence_mapping[prev_idx] = corr_id
                    else:
                        correspondence_lst.append(Correspondence(
                            prev_indices=[prev_idx],
                            curr_indices=[curr_matched_idx]
                        ))
                        corr_id = len(correspondence_lst) - 1
                        prev_correspondence_mapping[prev_idx] = corr_id
                        curr_correspondence_mapping[curr_matched_idx] = corr_id
                
                # case: no match found => pass
                else:
                    pass

            ## resolve split
            unassigned_curr = [i for i in range(len(curr_storms_map.storms)) if i not in curr_correspondence_mapping]
            for curr_idx in unassigned_curr:
                curr_pol = curr_storms_map.storms[curr_idx].contour
                prev_matched_idx = -1
                best_intersection_area = 0

                for prev_idx, prev_storm in enumerate(prev_storms_map.storms):
                    prev_pol = prev_storm.contour
                    if prev_pol.intersection(curr_pol).area > best_intersection_area:
                        best_intersection_area = prev_pol.intersection(curr_pol).area
                        prev_matched_idx = prev_idx
                
                # case: found the best match
                if best_intersection_area > 0:
                    if prev_matched_idx in prev_correspondence_mapping:
                        corr_id = prev_correspondence_mapping[prev_matched_idx]
                        correspondence_lst[corr_id].curr_indices.append(curr_idx)
                        curr_correspondence_mapping[curr_idx] = corr_id
                    else:
                        correspondence_lst.append(Correspondence(
                            prev_indices=[prev_matched_idx],
                            curr_indices=[curr_idx]
                        ))
                        corr_id = len(correspondence_lst) - 1
                        prev_correspondence_mapping[prev_matched_idx] = corr_id
                        curr_correspondence_mapping[curr_idx] = corr_id
                
                # case: no match found => create a new correspondence
                else:
                    correspondence_lst.append(Correspondence(
                        prev_indices=[],
                        curr_indices=[curr_idx]
                    ))
                    curr_correspondence_mapping[curr_idx] = len(correspondence_lst) - 1
            
            # re-order the storm list inside each correspondence
            ## idea: in case split => sort based on the disparity scores
            for corr in correspondence_lst:
                if len(corr.curr_indices) > 1:
                    # sort curr indices based on the disparity values
                    prev_idx = corr.prev_indices[0]
                    sorted_curr_indices = sorted(
                        corr.curr_indices,
                        key=lambda curr_idx: disparity_matrix[prev_idx, curr_idx]
                    )
                    corr.curr_indices = sorted_curr_indices
            
            self.tracker.update(correspondence_lst, prev_storms_map, curr_storms_map)

            # Update history movements to track history movement
            for storm in curr_storms_map.storms:
                storm_controller = self.tracker._get_track(storm.id)[0]
                storm.contour_color = storm_controller["storm_lst"][-1].contour_color if len(storm_controller) >= 1 else storm.contour_color
                storm.history_movements = [mv * dt for mv in storm_controller["movement"]]
            self.storms_maps.append(curr_storms_map)

        self.storms_maps.append(curr_storms_map)
        
        right_matches = list(set([curr for _, curr in assignments]))
        return len(right_matches)
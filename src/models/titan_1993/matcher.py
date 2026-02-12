import numpy as np
from shapely.geometry import Point

from src.tracking import BaseMatcher, BaseTrackingHistory
from src.cores.base import StormsMap

from src.cores.base import StormObject
from ..base.tracker import UpdateType, MatchedStormPair

class SimpleMatcher(BaseMatcher):
    max_velocity: float     # unit: pixel/hr

    def __init__(self, max_velocity: float):
        self.max_velocity = max_velocity

    def _construct_disparity_matrix(
            self, storm_lst1: list[StormObject], storm_lst2: list[StormObject]
        ) -> tuple[np.ndarray, np.ndarray]:
        # get square root of area difference
        area_lst1 = np.array([storm.contour.area for storm in storm_lst1])
        area_lst2 = np.array([storm.contour.area for storm in storm_lst2])
        area_matrix = np.sqrt(np.abs(area_lst1[:, None] - area_lst2[None, :]))

        # get centroid displacement
        centroid_lst1 = np.array([storm.centroid for storm in storm_lst1])
        centroid_lst2 = np.array([storm.centroid for storm in storm_lst2])
        centroid_displacement_matrix = np.linalg.norm(centroid_lst1[:,None,:] - centroid_lst2[None,:,:], axis=2)
        
        return area_matrix + centroid_displacement_matrix, centroid_displacement_matrix
    
    def match_storms(
            self, storm_map1: StormsMap, storm_map2: StormsMap
        ) -> np.ndarray:
        """
        Match storms between 2 time frame.

        Args:
            storm_map1 (StormsMap): storm map in the 1st frame.
            storm_map2 (StormsMap): storm map in the 2nd frame.
        
        Returns:
            assignments (np.ndarray): Array of (prev_idx, curr_idx) pairs representing matched storms.
        """
        assignments: list[MatchedStormPair] = []

        # Handle no storm found cases
        if len(storm_map2.storms) == 0:
            return []
        if len(storm_map1.storms) == 0:
            return [MatchedStormPair(
                prev_storm_order=-1,
                curr_storm_order=curr_idx,
                update_type=UpdateType.NEW
            ) for curr_idx in range(len(storm_map2.storms))]
        

        dt = (storm_map2.time_frame - storm_map1.time_frame).seconds / 3600     # unit: hr
        max_displacement = dt * self.max_velocity

        cost_matrix, displacement_matrix = self._construct_disparity_matrix(storm_map1.storms, storm_map2.storms)
        invalid_mask = displacement_matrix > max_displacement

        cost_matrix = cost_matrix + invalid_mask.astype(np.float64) * 2000      # add penalty to those violated
        row_ind, col_ind = self._hungarian_matching(cost_matrix)

        assignment_mask = np.zeros_like(invalid_mask, dtype=bool)
        assignment_mask[row_ind, col_ind] = True

        idx_pairs = np.argwhere(assignment_mask & np.logical_not(invalid_mask))

        # Update the history movement
        for prev_idx, curr_idx in idx_pairs:
            assignments.append(MatchedStormPair(
                prev_storm_order=prev_idx,
                curr_storm_order=curr_idx,
                update_type=UpdateType.MATCHED,
                estimated_movement=np.array([
                    storm_map2.storms[curr_idx].centroid[0] - storm_map1.storms[prev_idx].centroid[0],
                    storm_map2.storms[curr_idx].centroid[1] - storm_map1.storms[prev_idx].centroid[1]
                ])
            ))
        
        # resolve merge & split
        ## mapping: dict where key -> index of storm; value -> list of tuple[storm_id]
        mapping_prev = {int(prev_idx): [int(curr_idx)] for prev_idx, curr_idx in idx_pairs}
        mapping_curr = {int(curr_idx): [int(prev_idx)] for prev_idx, curr_idx in idx_pairs}

        ## Predict storms in storm_map1 to storm_map2 time_frame
        pred_storms_map = StormsMap([
            storm.forecast(dt=dt)
            for storm in storm_map1.storms
        ], time_frame=storm_map2.time_frame)

        # Check for merging
        for prev_idx in range(len(storm_map1.storms)):
            if prev_idx in mapping_prev:
                continue

            pred_storm = pred_storms_map.storms[prev_idx]

            # Find storms that the predicted centroid fall into.
            reversed_centroid = (pred_storm.centroid[1], pred_storm.centroid[0])    # (x, y) format for shapely
            candidates = [idx for idx, storm in enumerate(storm_map2.storms) \
                          if storm.contour.contains(Point(reversed_centroid))]
            
            # Case: more than 1 candidates => choose one with maximum overlapping on prev_storm
            if len(candidates) > 1:
                compute_overlapping = lambda pol: pred_storm.contour.intersection(pol).area / pred_storm.contour.area
                max_idx = np.argmax([compute_overlapping(storm_map2.storms[j].contour) for j in candidates])
                candidates = [candidates[max_idx]]
            
            if len(candidates) == 0:
                continue
            
            curr_idx = candidates[0]
            mapping_prev[prev_idx] = [curr_idx]
            centroid_displacement = np.array([
                storm_map2.storms[curr_idx].centroid[0] - storm_map1.storms[prev_idx].centroid[0],
                storm_map2.storms[curr_idx].centroid[1] - storm_map1.storms[prev_idx].centroid[1]
            ])
            if curr_idx not in mapping_curr:
                mapping_curr[curr_idx] = [prev_idx]
                assignments.append(MatchedStormPair(
                    prev_storm_order=prev_idx,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.MATCHED,
                    estimated_movement=centroid_displacement
                ))
            else:
                mapping_curr[curr_idx].append(prev_idx)
                assignments.append(MatchedStormPair(
                    prev_storm_order=prev_idx,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.MERGED,
                    estimated_movement=centroid_displacement
                ))

        # check for splitting
        for curr_idx in range(len(storm_map2.storms)):
            if curr_idx in mapping_curr:
                continue
            curr_storm = storm_map2.storms[curr_idx]
            # Find predicted storms that the current centroid fall into.
            reversed_centroid = (curr_storm.centroid[1], curr_storm.centroid[0])    # (x, y) format for shapely
            candidates = [
                    idx for idx, storm in enumerate(pred_storms_map.storms) \
                        if storm.contour.contains(Point(reversed_centroid))
                ]
            
            if len(candidates) == 0:
                assignments.append(MatchedStormPair(
                    prev_storm_order=-1,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.NEW
                ))
                continue

            # Case: more than 1 candidates => choose one with maximum overlapping on prev_storm
            if len(candidates) > 1:
                compute_overlapping = lambda pol: pol.contour.intersection(curr_storm.contour).area / pol.contour.area
                max_idx = np.argmax([compute_overlapping(pred_storms_map.storms[j]) for j in candidates])
                candidates = [candidates[max_idx]]
            
            prev_idx = candidates[0]
            mapping_curr[curr_idx] = [prev_idx]
            centroid_displacement = np.array([
                storm_map2.storms[curr_idx].centroid[0] - storm_map1.storms[prev_idx].centroid[0],
                storm_map2.storms[curr_idx].centroid[1] - storm_map1.storms[prev_idx].centroid[1]
            ])
            if prev_idx not in mapping_prev:
                mapping_prev[prev_idx] = [curr_idx]
                assignments.append(MatchedStormPair(
                    prev_storm_order=prev_idx,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.MATCHED,
                    estimated_movement=centroid_displacement
                ))
            else:
                mapping_prev[prev_idx].append(curr_idx)
                assignments.append(MatchedStormPair(
                    prev_storm_order=prev_idx,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.SPLITTED,
                    estimated_movement=centroid_displacement
                ))
        
        # fix split & merge motion estimation
        for prev_idx, curr_indices in mapping_prev.items():
            if len(curr_indices) <= 1:
                continue
            
            # compute area weights => weighted average
            area_list = [storm_map2.storms[curr_idx].contour.area for curr_idx in curr_indices]
            total_area = sum(area_list)
            area_weights = [area / total_area for area in area_list]

            # get motion assignments
            assignment_indices = [idx for idx, assign in enumerate(assignments) \
                                  if assign.prev_storm_order == prev_idx]
            movement_list = [assignments[idx].estimated_movement for idx in assignment_indices]

            # combined motion
            combined_motion = sum([
                movement_list[i] * area_weights[i]
                for i in range(len(curr_indices))
            ])
            
            # update all motions
            for idx in assignment_indices:
                assignments[idx].estimated_movement = combined_motion

        for curr_idx, prev_indices in mapping_curr.items():
            if len(prev_indices) <= 1:
                continue
            
            # compute area weights => weighted average
            area_list = [storm_map1.storms[prev_idx].contour.area for prev_idx in prev_indices]
            total_area = sum(area_list)
            area_weights = [area / total_area for area in area_list]

            # get motion assignments
            assignment_indices = [idx for idx, assign in enumerate(assignments) \
                                  if assign.curr_storm_order == curr_idx]
            movement_list = [assignments[idx].estimated_movement for idx in assignment_indices]

            # combined motion
            combined_motion = sum([
                movement_list[i] * area_weights[i]
                for i in range(len(prev_indices))
            ])
            
            # update all motions
            for idx in assignment_indices:
                assignments[idx].estimated_movement = combined_motion

        # Sort to ensure MATCHED are processed first
        assignments.sort(key=lambda x: x.update_type.value)
        return assignments
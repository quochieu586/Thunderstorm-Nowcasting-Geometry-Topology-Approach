import numpy as np
from shapely.geometry import box, Point

from src.cores.base import StormObject, StormsMap
from src.cores.movement_estimate.fft import FFTMovement
from src.tracking import BaseMatcher
from ..base.tracker import MatchedStormPair, UpdateType

class Matcher(BaseMatcher):
    fft: FFTMovement
    max_velocity: float
    max_velocity_diff: float
    max_cost: float
    
    def __init__(
            self, fft: FFTMovement, max_velocity: float, max_velocity_diff: float, max_cost: float
        ):
        self.max_velocity = max_velocity            # maximum allowed velocity (pixels/hour) for truncation
        self.max_velocity_diff = max_velocity_diff  # maximum difference between historical and estimated velocity
        self.max_cost = max_cost                    # maximum cost for cost matrix
        self.fft = fft

    def _construct_disparity_matrix(
            self, storm_lst1: list[StormObject], storm_lst2: list[StormObject], 
            shifts: np.ndarray
        ) -> np.ndarray:
        """
        Construct a disparity matrix based on centroids estimated from FFT.

        Args:
            storm_lst1 (list[CentroidStorm]): List of storms in previous time (target of matching).
            storm_lst2 (list[CentroidStorm]): List of storms in current time (candidate for matching).
            estimated_displacements (list[np.ndarray]): Estimated displacements for each storm in storm_lst1.
        
        Returns:
            np.ndarray: Disparity matrix.
        """
        targ_centroids = np.array([storm.centroid for storm in storm_lst1])
        cand_centroids = np.array([storm.centroid for storm in storm_lst2])
        pred_center = targ_centroids + shifts

        d_c = np.linalg.norm(
            targ_centroids[:, np.newaxis, :] - cand_centroids[np.newaxis, :, :], axis=-1
        )   # shape: (num_targ, num_cand)
        
        d_p = np.linalg.norm(
            pred_center[:, np.newaxis, :] - cand_centroids[np.newaxis, :, :], axis=-1
        )   # shape: (num_targ, num_cand)
        
        a_c = np.abs(
            np.array([storm.contour.area for storm in storm_lst1])[:, np.newaxis] - 
            np.array([storm.contour.area for storm in storm_lst2])[np.newaxis, :]
        )   # shape: (num_targ, num_cand)

        a_o = np.array([
            prev_storm.contour.intersection(curr_storm.contour).area
            for prev_storm in storm_lst1
            for curr_storm in storm_lst2
        ]).reshape(len(storm_lst1), len(storm_lst2))

        return np.clip(d_c + d_p + np.sqrt(a_c) - np.sqrt(a_o), 0, None)
    
    def _process_velocity(self, history_velocity: np.ndarray, estimated_velocity: np.ndarray) -> np.ndarray:
        """
        Process the velocity by combining historical and estimated velocities of a storm.

        Args:
            history_velocity (np.ndarray): Historical velocity of the storm.
            estimated_velocity (np.ndarray): Estimated velocity from FFT.

        Returns:
            np.ndarray: corrected velocity.
        """
        if history_velocity is None:
            return estimated_velocity
        elif np.linalg.norm(history_velocity - estimated_velocity) > self.max_velocity_diff:
            return estimated_velocity
        
        return 0.5 * (history_velocity + estimated_velocity)

    def match_storms(
            self, storms_map_1: StormsMap, storms_map_2: StormsMap
        ) -> list[MatchedStormPair]:
        """
        Match storms between 2 time frames.

        Args:
            storms_map_1 (StormsMap): Previous storms map.
            storms_map_2 (StormsMap): Current storms map.
            history_velocities (list[np.ndarray]): Previously recorded velocities for storms in storms_map_1.
        
        Returns:
            list[MatchedStormPair]: List of matched storm pairs.
        """
        # Step 1.1: Estimate movements via FFT
        fft_estimated_movements, region_list = self.fft.estimate_movement(
            storms_map_1, storms_map_2
        )                               # (Unit: pixels / hour)

        # Step 1.2: Get historical velocities
        history_velocities = []         # (Unit: pixels / hour)
        for storm in storms_map_1.storms:
            history_velocities.append(storm.get_movement())

        dt = (storms_map_2.time_frame - storms_map_1.time_frame).total_seconds() / 3600.0  # in hours

        corrected_shifts = np.array([
            self._process_velocity(history_velocity, estimated_velocity) * dt
            for history_velocity, estimated_velocity in zip(history_velocities, fft_estimated_movements)
        ])

        # step 2: Retrieve search regions = original regions + corrected shifts
        H, W = storms_map_1.dbz_map.shape
        regions = np.array(region_list)
        y_regions = regions[:, :2]
        x_regions = regions[:, 2:]
        y_search_regions = np.clip(y_regions + corrected_shifts[:, 0][:, np.newaxis], 0, H)
        x_search_regions = np.clip(x_regions + corrected_shifts[:, 1][:, np.newaxis], 0, W)

        search_regions = np.concatenate([y_search_regions, x_search_regions], axis=-1)

        # Step 3: Construct disparity matrix
        cost_matrix = self._construct_disparity_matrix(
            storms_map_1.storms,
            storms_map_2.storms,
            corrected_shifts
        )

        # step 3: Mark the valid matches => curr storm in the search region of previous storm
        valid_mask = np.zeros_like(cost_matrix, dtype=bool)

        for i in range(len(storms_map_1.storms)):
            min_y, max_y, min_x, max_x = search_regions[i]
            search_box = box(min_x, min_y, max_x, max_y)
            for j, storm in enumerate(storms_map_2.storms):
                if cost_matrix[i, j] > self.max_cost:
                    continue
                curr_pol = storm.contour
                # intersect a part or whole => valid
                if search_box.intersection(curr_pol).area > 0:
                    valid_mask[i, j] = True

        # step 4: Invalidate the invalid matches
        invalid_mask = ~valid_mask
        cost_matrix[invalid_mask] = 5000.0   # large cost for invalid matches

        # step 5: Find the optimal matching
        row_ind, col_ind = self._hungarian_matching(cost_matrix)

        assignment_mask = np.zeros_like(invalid_mask, dtype=bool)
        assignment_mask[row_ind, col_ind] = True

        idx_pairs = np.argwhere(assignment_mask & valid_mask)

        # update the history movement
        assignments: list[MatchedStormPair] = []
        for prev_idx, curr_idx in idx_pairs:
            assignments.append(MatchedStormPair(
                prev_storm_order=prev_idx,
                curr_storm_order=curr_idx,
                update_type=UpdateType.MATCHED,
                estimated_movement=corrected_shifts[prev_idx]   # (Unit: pixels)
            ))
        
        # resolve split & merge
        mapping_prev = {int(prev_idx): [int(curr_idx)] for prev_idx, curr_idx in idx_pairs}
        mapping_curr = {int(curr_idx): [int(prev_idx)] for prev_idx, curr_idx in idx_pairs}

        ## Predict storms in storm_map1 to storm_map2 time_frame
        pred_storms_map = StormsMap([
            storm.copy()
            for storm in storms_map_1.storms
        ], time_frame=storms_map_2.time_frame)

        # for storm, shift in zip(pred_storms_map.storms, corrected_shifts):
        #     storm.make_move(shift)

        ## Check for merging
        for prev_idx in range(len(storms_map_1.storms)):
            if prev_idx in mapping_prev:
                continue

            pred_storm = pred_storms_map.storms[prev_idx]

            # Find storms that the predicted centroid fall into.
            candidates = [idx for idx, storm in enumerate(storms_map_2.storms) if storm.contour.contains(Point(pred_storm.centroid))]
            
            # Case: more than 1 candidates => choose one with maximum overlapping on prev_storm
            if len(candidates) > 1:
                compute_overlapping = lambda pol: pred_storm.contour.intersection(pol).area / pred_storm.contour.area
                max_idx = np.argmax([compute_overlapping(storms_map_2.storms[j].contour) for j in candidates])
                candidates = [candidates[max_idx]]
            
            if len(candidates) == 0:
                continue
            
            curr_idx = candidates[0]
            mapping_prev[prev_idx] = [curr_idx]

            if curr_idx not in mapping_curr:
                mapping_curr[curr_idx] = [prev_idx]
                assignments.append(MatchedStormPair(
                    prev_storm_order=prev_idx,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.MATCHED,
                    estimated_movement=corrected_shifts[prev_idx]   # (Unit: pixels)
                ))
            else:
                mapping_curr[curr_idx].append(prev_idx)
                assignments.append(MatchedStormPair(
                    prev_storm_order=prev_idx,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.MERGED,
                    estimated_movement=corrected_shifts[prev_idx]   # (Unit: pixels)
                ))
        
        ## Check for splitting
        for curr_idx in range(len(storms_map_2.storms)):
            if curr_idx in mapping_curr:
                continue

            curr_storm = storms_map_2.storms[curr_idx]

            # Find storms that the current centroid fall into.
            candidates = [idx for idx, storm in enumerate(pred_storms_map.storms) if storm.contour.contains(Point(curr_storm.centroid))]
            
            # Case: more than 1 candidates => choose one with maximum overlapping on curr_storm
            if len(candidates) > 1:
                compute_overlapping = lambda pol: curr_storm.contour.intersection(pol).area / curr_storm.contour.area
                max_idx = np.argmax([compute_overlapping(pred_storms_map.storms[j].contour) for j in candidates])
                candidates = [candidates[max_idx]]
            
            if len(candidates) == 0:
                continue
            
            prev_idx = candidates[0]
            mapping_curr[curr_idx] = [prev_idx]

            if prev_idx not in mapping_prev:
                mapping_prev[prev_idx] = [curr_idx]
                assignments.append(MatchedStormPair(
                    prev_storm_order=prev_idx,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.MATCHED,
                    estimated_movement=corrected_shifts[prev_idx]   # (Unit: pixels)
                ))
            else:
                mapping_prev[prev_idx].append(curr_idx)
                assignments.append(MatchedStormPair(
                    prev_storm_order=prev_idx,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.SPLITTED,
                    estimated_movement=corrected_shifts[prev_idx]   # (Unit: pixels)
                ))

        # Finally, add NEW storms
        matched_curr_indices = set([pair.curr_storm_order for pair in assignments])
        for curr_idx in range(len(storms_map_2.storms)):
            if curr_idx in matched_curr_indices:
                continue
            assignments.append(MatchedStormPair(
                prev_storm_order=-1,
                curr_storm_order=curr_idx,
                update_type=UpdateType.NEW,
                estimated_movement=np.array([0.0, 0.0])   # (Unit: pixels)
            ))
        
        # Sort to ensure MATCHED are processed first
        assignments.sort(key=lambda x: x.update_type.value)
        return assignments
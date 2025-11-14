import numpy as np
from shapely.geometry import box

from src.tracking import BaseMatcher

from .storm import CentroidStorm
from .storm_map import DbzStormsMap

MAX_VELOCITY = 500          # pixels/hour
MAX_VELOCITY_DIFF = 500     # pixels/hour
MAX_COST = 45               # arbitrary units

class Matcher(BaseMatcher):
    max_velocity: float
    max_velocity_diff: float

    def __init__(self, max_velocity: float = MAX_VELOCITY, max_velocity_diff: float = MAX_VELOCITY_DIFF, max_cost: float = MAX_COST):
        self.max_velocity = max_velocity            # maximum allowed velocity (pixels/hour) for truncation
        self.max_velocity_diff = max_velocity_diff  # maximum difference between historical and estimated velocity
        self.max_cost = max_cost                    # maximum cost for cost matrix

    def _construct_disparity_matrix(
            self, storm_lst1: list[CentroidStorm], storm_lst2: list[CentroidStorm], 
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
            self, storms_map_1: DbzStormsMap, storms_map_2: DbzStormsMap, history_velocities: list[np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Match storms between 2 time frames.

        Args:
            storms_map_1 (DbzStormsMap): Previous storms map.
            storms_map_2 (DbzStormsMap): Current storms map.
            history_velocities (list[np.ndarray]): Previously recorded velocities for storms in storms_map_1.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: 
                + matched indices (N x 2) where N is the number of matched storm pairs.
                + cost matrix (num_storms_1 x num_storms_2).
        """
        # region_list (list[min_y, min_x, max_y, max_x]): regions for each storm without the buffer
        region_list = storms_map_1.fft_estimate(storms_map_2, max_velocity=self.max_velocity)
        dt = (storms_map_2.time_frame - storms_map_1.time_frame).total_seconds() / 3600.0  # in hours
        max_displacement = self.max_velocity * dt

        # Step 1: Retrieve displacements
        estimated_velocities = [
            np.array(storm.estimated_velocity) for storm in storms_map_1.storms
        ]

        corrected_shifts = np.array([
            self._process_velocity(history_velocity, estimated_velocity) * dt
            for history_velocity, estimated_velocity in zip(history_velocities, estimated_velocities)
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

        return np.argwhere(assignment_mask & valid_mask), cost_matrix
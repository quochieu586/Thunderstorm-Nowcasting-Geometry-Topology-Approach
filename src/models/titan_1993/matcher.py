import numpy as np

from src.tracking import BaseMatcher, BaseTrackingHistory
from src.cores.base import StormsMap

from .storm import CentroidStorm

MAX_VELOCITY = 500

class SimpleMatcher(BaseMatcher):
    max_velocity: float     # unit: pixel/hr

    def __init__(self, max_velocity: float = MAX_VELOCITY):
        self.max_velocity = max_velocity

    def _construct_disparity_matrix(
            self, storm_lst1: list[CentroidStorm], storm_lst2: list[CentroidStorm]
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
        dt = (storm_map2.time_frame - storm_map1.time_frame).seconds / 3600     # unit: hr
        max_displacement = dt * self.max_velocity

        cost_matrix, displacement_matrix = self._construct_disparity_matrix(storm_map1.storms, storm_map2.storms)
        invalid_mask = displacement_matrix > max_displacement

        cost_matrix = cost_matrix + invalid_mask.astype(np.float64) * 2000      # add penalty to those violated
        row_ind, col_ind = self._hungarian_matching(cost_matrix)

        assignment_mask = np.zeros_like(invalid_mask, dtype=bool)
        assignment_mask[row_ind, col_ind] = True

        return np.argwhere(assignment_mask & np.logical_not(invalid_mask))
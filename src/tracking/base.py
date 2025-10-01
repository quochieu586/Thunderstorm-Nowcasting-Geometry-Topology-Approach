from abc import ABC, abstractmethod
from typing import Optional
from scipy.optimize import linear_sum_assignment
import numpy as np

from src.cores.contours import StormObject, StormsMap

class BaseTracker(ABC):
    """
    Base class for storm tracking. In tracking steps, we match detected storms from two consecutive time frames
    """

    @abstractmethod
    def _cost_function(self, storm1: StormObject, storm2: StormObject) -> float:
        """
        Calculate the cost of associating two storm objects.
        
        Parameters:
            storm1 (StormObject): The first storm object.
            storm2 (StormObject): The second storm object.
        
        Returns:
            float: The cost of associating the two storm objects. This cost must range from 0 to 1, where 1 indicates a perfect match and 0 indicates absolutely no matching.
        """
        pass

    def construct_disparity_matrix(
            self, storms_map1: StormsMap, storms_map2: StormsMap, include_matches: bool = True, cancel_threshold: float = 0.1
        ) -> tuple[np.ndarray, Optional[list[tuple[int, int]]]]:
        """
        Construct a disparity matrix for the given storm maps.

        Parameters:
            storms_map1 (StormsMap): The first set of storm objects.
            storms_map2 (StormsMap): The second set of storm objects.
            include_matches (bool): Whether to include matched storm pairs in the output using Hungarian algorithm.
            cancel_threshold (float): The threshold below which matches are considered invalid.

        Returns:
            cost_matrix (np.ndarray): The constructed disparity matrix.
            matches (list of tuples): The list of matched storm pairs (index in storms_map1, index in storms_map2).
        """
        num_storms1 = len(storms_map1.storms)
        num_storms2 = len(storms_map2.storms)

        if num_storms1 == 0 or num_storms2 == 0:
            return np.array([]), None

        cost_matrix = np.zeros((num_storms1, num_storms2))

        for i, storm1 in enumerate(storms_map1.storms):
            for j, storm2 in enumerate(storms_map2.storms):
                cost_matrix[i, j] = self._cost_function(storm1, storm2)
        
        if not include_matches:
            return cost_matrix, None
        
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        matches = [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] > cancel_threshold]
        return cost_matrix, matches
    
    def hungarian_matching(self, cost_matrix: np.ndarray, cancel_threshold: float = 0.1) -> list[tuple[int, int]]:
        """
        Perform Hungarian matching on the given cost matrix.

        Parameters:
            cost_matrix (np.ndarray): The cost matrix.
            cancel_threshold (float): The threshold below which matches are considered invalid.

        Returns:
            matches (list of tuples): The list of matched storm pairs (index in first set, index in second set).
        """
        if cost_matrix.size == 0:
            return []
        
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        matches = [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] > cancel_threshold]
        return matches
from abc import ABC, abstractmethod
from src.cores.base import BaseObject
import numpy as np
from scipy.optimize import linear_sum_assignment

class BaseMatcher(ABC):
    """
    Base class for matching between consecutive frames.
    """
    @abstractmethod
    def _construct_disparity_matrix(
        self, object_lst1: list[BaseObject], object_lst2: list[BaseObject]
    ) -> np.ndarray:
        """
        Construct a disparity matrix for the given storm maps.

        Args:
            object_lst1 (list[BaseObject]): The first list of objects.
            object_lst2 (list[BaseObject]): The second list of objects.

        Returns:
            cost_matrix (np.ndarray): The constructed disparity matrix.
        """
        pass

    def _hungarian_matching(self, cost_matrix: np.ndarray) -> np.ndarray:
        """
        Find an optimal one-to-one assignment.

        Args:
            cost_matrix: (np.ndarray): The constructed disparity matrix.

        Returns:
            assignments (tuple[np.ndarray]): The list of x-, y- assignments.
        """
        return linear_sum_assignment(cost_matrix)

class BaseTrackingHistory(ABC):
    """
    Track the history of storms.
    """
    tracks: list[dict]  # list of track
    storm_dict: dict    # key -> id of current storm, val -> id of corresponding track
    active_list = list  # keep the id of current active storms

class BaseTracker(ABC):
    """
    Track storms over time.
    """
    matcher: BaseMatcher
    tracker: BaseTrackingHistory
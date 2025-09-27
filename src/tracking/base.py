# from abc import abstractmethod, ABC

# from scipy.optimize import linear_sum_assignment
# import numpy as np

# from src.contours import StormObject, StormsMap

# class BaseStormTracker:
#     storms_map: list[StormsMap]

#     def __init__(self):
#         self.storms_map = []

#     def track_storms(self, detected_storms: list[StormObject], previous_storms: list[StormObject]) -> list[StormObject]:
#         """
#             Track storms across frames using the Hungarian algorithm for optimal assignment.

#             Args:
#                 detected_storms (List[StormObject]): The list of detected storms in the current frame.
#                 previous_storms (List[StormObject]): The list of storms from the previous frame.

#             Returns:
#                 List[StormObject]: The list of tracked storms.
#         """
#         if not detected_storms or not previous_storms:
#             return detected_storms

#         # Create cost matrix
#         cost_matrix = self._create_cost_matrix(detected_storms, previous_storms)

#         # Solve assignment problem
#         row_indices, col_indices = linear_sum_assignment(cost_matrix)

#         # Map detected storms to previous storms
#         tracked_storms = []
#         for row, col in zip(row_indices, col_indices):
#             if cost_matrix[row, col] < self.tracking_threshold:
#                 tracked_storms.append(detected_storms[row])
#             else:
#                 tracked_storms.append(None)

#         return tracked_storms

#     def _create_cost_matrix(self,  ) -> np.ndarray:
#         """
#             Create a cost matrix for the Hungarian algorithm.

#             Args:
#                 detected_storms (List[StormObject]): The list of detected storms in the current frame.
#                 previous_storms (List[StormObject]): The list of storms from the previous frame.

#             Returns:
#                 np.ndarray: The cost matrix.
#         """
#         cost_matrix = np.zeros((len(detected_storms), len(previous_storms)))

#         for i, detected in enumerate(detected_storms):
#             for j, previous in enumerate(previous_storms):
#                 cost_matrix[i, j] = self._compute_cost(detected, previous)

#         return cost_matrix

#     @
#     def _compute_cost(self, detected: StormObject, previous: StormObject) -> float:
#         """
#             Compute the cost of assigning a detected storm to a previous storm.

#             Args:
#                 detected (StormObject): The detected storm.
#                 previous (StormObject): The previous storm.

#             Returns:
#                 float: The computed cost.
#         """
#         # Implement your cost function here
#         return np.linalg.norm(detected.centroid - previous.centroid)
    
#     def matching_pairs(self, dectected)

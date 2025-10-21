import numpy as np
from shapely.affinity import translate
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from src.tracking import BaseMatcher
from src.cores.metrics import overlapping_storm_area
from src.cores.base import StormsMap

from .storm import StormShapeVectors

RADII = [20, 40, 60, 80, 100, 120]
NUM_SECTORS = 8
CIRCLE_AREA = RADII[-1] * RADII[-1] * np.pi
CLUSTER_COUNT = 2

class PolarVectorMatcher(BaseMatcher):
    def __init__(self, discard_threshold: float = 0.2):
        self.discard_threshold = discard_threshold

    def _compute_similarity_pair(self, vector_1: np.ndarray, vector_2: np.ndarray, circle_area: float):
        return np.linalg.norm((vector_1 - vector_2)) / circle_area

    def _compute_valid_pairs(self, prev_storm: StormShapeVectors, curr_storm: StormShapeVectors, circle_area: float, 
                             maximum_displacement: float = 10, thresholds: list[float] = [0.05, 0.08, 0.1]) -> list[tuple[float, float]]:
        """
            Compute movement between two lists of shape vectors. For all pairs of points between two storms, if the similarity score is below a certain threshold and the distance between two points is below a certain maximum displacement, we consider this pair as a valid movement. If the number of valid movements exceeds 20% of the number of vertices, we return this list of movements.

            Args:
                prev_storm (Storm): Previous storm instance.
                curr_storm (Storm): Current storm instance.
                circle_area (float): Area of the circle used in shape vector computation.
                maximum_displacement (float): Maximum allowed displacement between two points.
                thresholds (list[float]): List of ascending thresholds for similarity scores.

            Returns:
                List[Tuple[float, float]]: List of valid displacements.
        """
        if prev_storm.contour.distance(curr_storm.contour) > maximum_displacement:
            return []

        num_vertices = max(len(prev_storm.shape_vectors), len(curr_storm.shape_vectors))
        displacements = [[]] * len(thresholds)

        for svector_1 in prev_storm.shape_vectors:
            for svector_2 in curr_storm.shape_vectors:
                if np.linalg.norm(np.array(svector_1.coord) - np.array(svector_2.coord)) > maximum_displacement:
                    continue
                score = self._compute_similarity_pair(svector_1.vector, svector_2.vector, circle_area)

                for idx, threshold in enumerate(thresholds):
                    if score < threshold:
                        v1, v2 = svector_1.coord, svector_2.coord
                        displacements[idx].append((v2[0]-v1[0], v2[1]-v1[1]))
        
        # Only return the the matching pairs at the first threshold that has enough valid pairs
        for idx in range(len(thresholds)):
            if len(displacements[idx]) > (num_vertices / 5):
                return displacements[idx]
        
        return []

    def _get_translate_vector(self, displacement_list: list[tuple[float, float]], cluster_counts: int=CLUSTER_COUNT) -> tuple[float, float, np.ndarray]:
        """
        From list of displacements, use K-means clustering to find the largest cluster and return its center as the overall translation vector.
        Return the translation vector (x_offset, y_offset) and the labels of each displacement point for visualization.
        """
        if len(displacement_list) == 0:
            return 0.0, 0.0, np.array([])
        kmeans = KMeans(n_clusters=CLUSTER_COUNT)
        kmeans.fit(displacement_list)
        centers = kmeans.cluster_centers_
        cluster_counts = np.bincount(kmeans.labels_)
        max_cluster_idx = np.argmax(cluster_counts)

        translate_vector = centers[max_cluster_idx]
        return translate_vector[0], translate_vector[1], kmeans.labels_

    def _cost_function(self, prev_contour: StormShapeVectors, curr_contour: StormShapeVectors) -> float:
        displacement_list = self._compute_valid_pairs(prev_contour, curr_contour, CIRCLE_AREA)
        if len(displacement_list) == 0:
            return 0.0
        xoff, yoff, _ = self._get_translate_vector(displacement_list, cluster_counts=CLUSTER_COUNT)
        predicted_polygon = translate(prev_contour.contour, xoff=xoff, yoff=yoff)
        overlapping = overlapping_storm_area(predicted_polygon, curr_contour.contour)

        return overlapping
    
    def _construct_disparity_matrix(
            self, storm_lst1: list[StormShapeVectors], storm_lst2: list[StormShapeVectors]
        ) -> np.ndarray:
        """
        Construct the cost matrix for Hungarian matching.
        """
        # get square root of area difference
        disparity_matrix = np.zeros((len(storm_lst1), len(storm_lst2)))

        for i, prev_storm in enumerate(storm_lst1):
            for j, curr_storm in enumerate(storm_lst2):
                disparity_matrix[i, j] = -self._cost_function(prev_storm, curr_storm)

        return disparity_matrix
    
    def match_storms(self, storm_1: StormsMap, storm_2: StormsMap) -> list[tuple[int, int]]:
        """
        Match storms between 2 time frame.

        Args:
            storm_map_1 (StormsMap): storm map in the 1st frame.
            storm_map_2 (StormsMap): storm map in the 2nd frame.
        
        Returns:
            assignments (np.ndarray): Array of (prev_idx, curr_idx) pairs representing matched storms.
        """
        disparity_matrix = self._construct_disparity_matrix(storm_1.storms, storm_2.storms)
        prev_indices, curr_indices = linear_sum_assignment(disparity_matrix)

        assignments = [(prev_idx, curr_idx) for prev_idx, curr_idx in zip(prev_indices, curr_indices) if -disparity_matrix[prev_idx, curr_idx] > self.discard_threshold]

        return assignments

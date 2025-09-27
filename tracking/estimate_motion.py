import numpy as np
from typing import List, Tuple
import math
from dataclasses import dataclass
from sklearn.cluster import KMeans
from shapely import Polygon
from shapely.affinity import translate

from .process_polygon import compute_overlapping

Coord = Tuple[float, float]

@dataclass
class ShapeVector:
    coord: Tuple[float, float]
    vector: np.ndarray

@dataclass
class Storm:
    pol: Polygon
    vectors: List[ShapeVector]

def _compute_similarity_pair(svector_1: ShapeVector, svector_2: ShapeVector, circle_area: float):
    """
        Compute similarity between two shape vectors.
    """
    return np.sum(np.abs(svector_1.vector - svector_2.vector)) / circle_area

def _compute_similarity_pair(svector_1: ShapeVector, svector_2: ShapeVector, circle_area: float):
    """
        Compute similarity between two shape vectors.
    """
    return np.sum(np.abs(svector_1.vector - svector_2.vector)) / circle_area

def compute_valid_pairs(
        storm_1: Storm,
        storm_2: Storm,
        circle_area: float,
        maximum_displacement: float = 20,
        thresholds: List[float] = [0.05, 0.08, 0.1]
    ) -> List[List[Tuple[float, Tuple[float, float]]]]:
    """
        Compute movement between two lists of shape vectors.

        Args:
            shape_polygon_1 (List[ShapeVector]): List of shape vectors in the first frame.
            shape_polygon_2 (List[ShapeVector]): List of shape vectors in the second frame.
            circle_area (float): Area of the circle used for normalization.
            thresholds (float): List of ascending threshold

        Returns:
            List[Tuple[float, float]]: List of valid displacements.
    """
    if storm_1.pol.distance(storm_2.pol) > maximum_displacement:
        return []

    num_vertices = max(len(storm_1.vectors), len(storm_2.vectors))
    displacements = [[]] * len(thresholds)
    for svector_1 in storm_1.vectors:
        for svector_2 in storm_2.vectors:
            if compute_distance(svector_1, svector_2) > maximum_displacement:
                continue
            score = _compute_similarity_pair(svector_1, svector_2, circle_area)

            for idx, threshold in enumerate(thresholds):
                if score < threshold:
                    v1, v2 = svector_1.coord, svector_2.coord
                    displacements[idx].append((v2[0]-v1[0], v2[1]-v1[1]))
    
    for idx in range(len(thresholds)):
        if len(displacements[idx]) > (num_vertices / 5):
            # print(f"Return num of valid pairs at threshold {thresholds[idx]}")
            return displacements[idx]
    
    return []

def generate_final_displacement(
        displacement_list: List[Tuple[float, Tuple[float, float]]], alpha: float = 1
    ) -> Tuple[float, float]:
    """
    Args:
        displacement_list (List[Tuple[float, Tuple[float, float]]]): List of displacement with weight
    Returns:
        dx, dy (Tuple[float, float])
    """
    weights = [np.exp(w * alpha) for w, _ in displacement_list]
    sum_w = sum(weights)
    weights = [w / sum_w for w in weights]
    
    final_dx = sum([dx * w for w, (_, (dx, _)) in zip(weights, displacement_list)])
    final_dy = sum([dy * w for w, (_, (_, dy)) in zip(weights, displacement_list)])
    return final_dx, final_dy

def estimate_displacement(
        prev_storm: Storm,
        curr_storms: List[Storm],
        circle_area: float,
        maximum_displacement: float = 20,
        thresholds: List[float] = [0.05, 0.08, 0.1],
        alpha_smoothing: float = 5
    ) -> Tuple[float, float]:
    displacement_list_storm = []
    for storm in curr_storms:
        displacements = compute_valid_pairs(
                prev_storm, storm, circle_area, maximum_displacement, thresholds = thresholds
            )

        if len(displacements) == 0:
            continue
        
        best_overlapping_score = 0
        best_displacement = None

        for num_clusters in range(1, 20, 1):
            try:
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(displacements)

                centers = kmeans.cluster_centers_
                cluster_counts = np.bincount(kmeans.labels_)

                max_cluster_idx = np.argmax(cluster_counts)

                predicted_polygon = translate(prev_storm.pol, xoff=centers[max_cluster_idx][0], yoff=centers[max_cluster_idx][1])
                overlapping = compute_overlapping([predicted_polygon], [storm.pol])[0][0]

                if overlapping > best_overlapping_score:
                    best_overlapping_score = overlapping
                    best_displacement = centers[max_cluster_idx]
            except:
                break
        
        if best_overlapping_score == 0:
            continue
        
        displacement_list_storm.append((best_overlapping_score, best_displacement))
    
    return generate_final_displacement(displacement_list_storm, alpha=alpha_smoothing)

def compute_distance(v1: ShapeVector, v2: ShapeVector) -> float:
    """
        Compute Euclidean distance between two shape vectors.
    """
    x1, y1 = v1.coord
    x2, y2 = v2.coord

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# def _estimate_movement(
#         movements: List[Tuple[float, Tuple[Coord, Coord]]],
#         temperature: float = 1.0
#     ) -> Tuple[Coord, Coord]:
#     """
#         Estimate movement using softmax-weighted average of displacements.

#         Args:
#             movements (List[Tuple[float, Tuple[Coord, Coord]]]): List of tuples (score, (v1, v2)) 
#                 where v1 and v2 are coordinates.
#             temperature (float): Temperature parameter for softmax.
        
#         Returns:
#             Estimated displacement as a tuple (dx, dy).
#     """
#     if len(movements) == 0:
#         return (0 ,0)

#     if len(movements) == 1:
#         v1, v2 = movements[0][1]
#         return (v2[0] - v1[0], v2[1] - v1[1])
    
#     # Extract scores and displacements
#     scores = np.array([score for score, _ in movements])
#     displacements = [(v2[0] - v1[0], v2[1] - v1[1]) for _, (v1, v2) in movements]

#     # Apply softmax to scores
#     exp_scores = np.exp(scores / temperature)
#     weights = exp_scores / np.sum(exp_scores)

#     # Weighted average displacement
#     dx, dy = np.sum(weights[:, None] * displacements, axis=0)

#     return (float(dx), float(dy))

# def check_valid_movement(point_1: Coord, point_2: Coord, area: float) -> bool:
#     """
#         Check valid movement between two points.
#     """
#     return True

# def compute_movement(
#         shape_polygon_1: List[ShapeVector], 
#         shape_polygon_2: List[ShapeVector], 
#         circle_area: float, 
#         threshold: float = 0.05
#     ) -> Tuple[Tuple[float, float], float]:
#     """
#         Compute movement between two lists of shape vectors.

#         Args:
#             shape_polygon_1 (List[ShapeVector]): List of shape vectors in the first frame.
#             shape_polygon_2 (List[ShapeVector]): List of shape vectors in the second frame.
#             circle_area (float): Area of the circle used for normalization.
#             threshold (float): Similarity threshold to consider a pair as valid.

#         Returns:
#             Tuple[Tuple[float, float], float]: Estimated displacement (dx, dy) and best similarity score.
#     """
#     valid_pairs = []
#     best_score = float("inf")

#     for svector_1 in shape_polygon_1:
#         for svector_2 in shape_polygon_2:
#             score = _compute_similarity_pair(svector_1, svector_2, circle_area)
#             if score < threshold:
#                 valid_pairs.append((score, (svector_1.coord, svector_2.coord)))
            
#             if score < best_score:
#                 best_score = score
                
#     return best_score, _estimate_movement(valid_pairs)
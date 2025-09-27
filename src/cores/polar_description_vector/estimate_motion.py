import numpy as np
from typing import List, Tuple
import math
from dataclasses import dataclass

Coord = Tuple[float, float]

@dataclass
class ShapeVector:
    coord: Tuple[float, float]
    vector: np.ndarray

def _compute_similarity_pair(svector_1: ShapeVector, svector_2: ShapeVector, circle_area: float):
    """
        Compute similarity between two shape vectors.
    """
    return np.sum(np.abs(svector_1.vector - svector_2.vector)) / circle_area

def _estimate_movement(
        movements: List[Tuple[float, Tuple[Coord, Coord]]],
        temperature: float = 1.0
    ) -> Tuple[Coord, Coord]:
    """
        Estimate movement using softmax-weighted average of displacements.

        Args:
            movements (List[Tuple[float, Tuple[Coord, Coord]]]): List of tuples (score, (v1, v2)) 
                where v1 and v2 are coordinates.
            temperature (float): Temperature parameter for softmax.
        
        Returns:
            Estimated displacement as a tuple (dx, dy).
    """
    if len(movements) == 0:
        return (0 ,0)

    if len(movements) == 1:
        v1, v2 = movements[0][1]
        return (v2[0] - v1[0], v2[1] - v1[1])
    
    # Extract scores and displacements
    scores = np.array([score for score, _ in movements])
    displacements = [(v2[0] - v1[0], v2[1] - v1[1]) for _, (v1, v2) in movements]

    # Apply softmax to scores
    exp_scores = np.exp(scores / temperature)
    weights = exp_scores / np.sum(exp_scores)

    # Weighted average displacement
    dx, dy = np.sum(weights[:, None] * displacements, axis=0)

    return (float(dx), float(dy))

def check_valid_movement(point_1: Coord, point_2: Coord, area: float) -> bool:
    """
        Check valid movement between two points.
    """
    return True

def compute_movement(
        shape_polygon_1: List[ShapeVector], 
        shape_polygon_2: List[ShapeVector], 
        circle_area: float, 
        threshold: float = 0.05
    ) -> Tuple[Tuple[float, float], float]:
    """
        Compute movement between two lists of shape vectors.

        Args:
            shape_polygon_1 (List[ShapeVector]): List of shape vectors in the first frame.
            shape_polygon_2 (List[ShapeVector]): List of shape vectors in the second frame.
            circle_area (float): Area of the circle used for normalization.
            threshold (float): Similarity threshold to consider a pair as valid.

        Returns:
            Tuple[Tuple[float, float], float]: Estimated displacement (dx, dy) and best similarity score.
    """
    valid_pairs = []
    best_score = float("inf")

    for svector_1 in shape_polygon_1:
        for svector_2 in shape_polygon_2:
            score = _compute_similarity_pair(svector_1, svector_2, circle_area)
            if score < threshold:
                valid_pairs.append((score, (svector_1.coord, svector_2.coord)))
            
            if score < best_score:
                best_score = score
                
    return best_score, _estimate_movement(valid_pairs)

def compute_distance(v1: ShapeVector, v2: ShapeVector) -> float:
    """
        Compute Euclidean distance between two shape vectors.
    """
    x1, y1 = v1.coord
    x2, y2 = v2.coord

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
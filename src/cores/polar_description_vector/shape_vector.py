import numpy as np
from dataclasses import dataclass
from shapely.geometry import Polygon

from .process_polygon import construct_shape_vector, simplify_contour

@dataclass
class ShapeVector:
    coord: tuple[float, float]
    vector: np.ndarray

class StormShapeVectors:
    """
    Enpoints for saving and extracting shape vectors around the contour of a storm.
    """
    contour: np.ndarray
    shape_vectors: list[ShapeVector]
    coords: np.ndarray

    def __init__(self, contour: np.ndarray):
        self.contour = contour
        self.coords = simplify_contour(contour)
        self.shape_vectors = []

    def extract_shape_vectors(self, global_contours: list[Polygon], radii = [20, 40, 60], num_sectors = 8):
        """
            Extract shape vectors around each vertex of the contour.

            Args:
                radii: a list of radius of sectors (default: [20, 40, 60]).
                num_sectors: number of sectors (default: 8).

            Returns:
                List of shape vectors.
        """
        vectors = [
            construct_shape_vector(polygons=global_contours, point=coord, radii=radii, num_sectors=num_sectors)
            for coord in self.coords
        ]
        self.shape_vectors = [ShapeVector(coord=(coord[0], coord[1]), vector=vector) for coord, vector in zip(self.coords.reshape(-1, 2), vectors)]
        return self.shape_vectors
    
    def similarity_score(self, other: 'StormShapeVectors'):
        """
            Compute similarity score between two sets of shape vectors. Methods:
            ....
            TODO: implement later.

            Args:
                other: another StormShapeVectors instance.

            Returns:
                Similarity score (float).
        """
        if not self.shape_vectors or not other.shape_vectors:
            raise ValueError("Both shape vector lists must be non-empty.")

        total_score = 0.0
        count = 0

        for sv1 in self.shape_vectors:
            for sv2 in other.shape_vectors:
                if sv1.vector.size == 0 or sv2.vector.size == 0:
                    continue
                score = np.exp(-np.linalg.norm(sv1.vector - sv2.vector))
                total_score += score
                count += 1

        return total_score / count if count > 0 else 0.0
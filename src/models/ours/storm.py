import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.affinity import translate
from sklearn.cluster import KMeans
from copy import deepcopy

from src.cores.base import StormObject
from src.cores.polar_description_vector import ShapeVector, construct_shape_vector_fast
from src.preprocessing import convert_polygons_to_contours

THRESHOLD = 35
DISTANCE_DBZ = 5
FILTER_AREA = 20        # storm with area under this threshold => cancel
FILTER_CENTER = 10

# Implementation of storm object
RADII = [20, 40, 60, 80, 100, 120]
NUM_SECTORS = 8
DENSITY = 0.05


class ShapeVectorStorm(StormObject):
    shape_vectors: list[ShapeVector]
    # coords: np.ndarray

    def __init__(
            self, polygon: Polygon, global_contours: list[np.ndarray], img_shape: tuple[int, int], id: str = "",
            density: float = DENSITY, radii: list[float] = RADII, num_sectors: int = NUM_SECTORS
        ):
        # initialize with the contour and the id
        super().__init__(contour=polygon, id=id)
        self.img_shape = img_shape
        
        contour = convert_polygons_to_contours([self.contour])[0]
        coords = self._sample_particles(contour, density)

        # create the shape vectors
        vectors = construct_shape_vector_fast(
            global_contours=global_contours, particles=coords, num_sectors=num_sectors, radii=radii
        )
        self.shape_vectors = [ShapeVector(
            coord=(coord[0], coord[1]), vector=vector
        ) for coord, vector in zip(coords.reshape(-1, 2), vectors)]
    
    def copy(self) -> "ShapeVectorStorm":
        # Allocate new object without calling __init__
        new_obj = self.__class__.__new__(self.__class__)

        # Copy all attributes manually (deepcopy for safety)
        new_obj.__dict__ = deepcopy(self.__dict__)
        new_obj.id = "pred_" + new_obj.id

        return new_obj
    
    def make_move(self, displacement: np.ndarray):
        dx, dy = displacement
        self.contour = translate(self.contour, xoff=dx, yoff=dy)
        for vector in self.shape_vectors:
            vector.coord = vector.coord[0] + dx, vector.coord[1] + dy
    
    def get_num_particles(self):
        """
        Get the number of particles.
        """
        return len(self.shape_vectors)
    
    def _sample_particles(self, contour: np.ndarray, density: float) -> np.ndarray:
        """
        Sample a list of particles inside the contour.

        Args:
            contour (np.ndarray): the list of points, in order, creating the contour.
            div (float, default): determine the number of particles = area / div.
        
        Returns:
            particles (np.ndarray): the list of particles.
        """
        # get the set of points bounded by the contour
        mask = np.zeros(shape=self.img_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], color=1)
        points = np.argwhere(mask > 0)

        # cluster them 
        n_clusters = int(cv2.contourArea(contour) * density) + 1
        k_means = KMeans(n_clusters, random_state=2025)
        k_means.fit(points)

        return k_means.cluster_centers_.astype(np.int64)[:, ::-1]   # revert the coord order.
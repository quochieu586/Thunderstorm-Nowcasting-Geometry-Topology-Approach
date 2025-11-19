import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from sklearn.cluster import KMeans
from copy import deepcopy

from src.cores.base import StormObject
from src.cores.polar_description_vector import ShapeVector, construct_shape_vector_fast, construct_sector
from src.preprocessing import convert_polygons_to_contours
import rasterio.features as rfeat

THRESHOLD = 35
DISTANCE_DBZ = 5
FILTER_AREA = 20        # storm with area under this threshold => cancel
FILTER_CENTER = 10

# Implementation of storm object
RADII = [20, 40, 60, 80, 100, 120]
NUM_SECTORS = 8
DENSITY = 0.05


def scale_map(dbz_map: np.ndarray) -> np.ndarray:
    """
    Scale image to (0,1) ranges where min considered dbz is 20 and max is 65 dBZ
    """
    # scaled_map = (dbz_map - 20) / (65 - 20)
    # scaled_map = np.clip(scaled_map, 0, 1)
    # return scaled_map
    return (dbz_map >= 35).astype(np.float32)

def shift_mask(mask, xoff, yoff, origin=(200, 200)):
    H, W = mask.shape
    dx = xoff - origin[0]
    dy = yoff - origin[1]
    
    out = np.zeros_like(mask)
    
    # compute valid ranges
    x1 = max(0, dx)
    x2 = min(W, W + dx)
    y1 = max(0, dy)
    y2 = min(H, H + dy)
    
    out[y1:y2, x1:x2] = mask[y1-dy:y2-dy, x1-dx:x2-dx]
    return out

def _construct_shape_vector_dbz(
        dbz_map: np.ndarray, particles: np.ndarray, shape: tuple,
        num_sectors: int = NUM_SECTORS, radii: list = RADII
    ) -> np.ndarray:
    """
    Construct shape vectors for given particles based on dbz map.
    """
    # scale the dbz map to (0,1)
    scaled_map = scale_map(dbz_map)
    
    # prepare sector masks
    origin = (radii[-1], radii[-1])

    sector_mask_templates = []
    prev_base_sector = None
    for radius in RADII:
        new_sector = construct_sector(origin, radius, 0, 360/NUM_SECTORS)
        if prev_base_sector is not None:
            base_sector = new_sector.difference(prev_base_sector)
        else:
            base_sector = new_sector

        prev_base_sector = new_sector

        for j in range(num_sectors):
            sector = rotate(base_sector, j*360/num_sectors, origin=origin)
            sector_mask_templates.append(rfeat.rasterize([(sector, 1)], out_shape=shape, dtype=np.float32))

    sector_mask_templates = np.stack(sector_mask_templates, axis=-1)
    
    # construct shape vectors => shape: (num_particles, num_sectors * num_radii)
    shape_vectors = np.zeros((particles.shape[0], len(radii) * num_sectors), dtype=np.float32)

    for p_idx, (xoff, yoff) in enumerate(particles):
        shape_vector = np.zeros((len(radii) * num_sectors,), dtype=np.float32)

        for i in range(len(radii) * num_sectors):
            sector_mask = shift_mask(sector_mask_templates[:, :, i], xoff, yoff, origin)
            shape_vector[i] = np.sum(sector_mask * scaled_map)

        shape_vectors[p_idx, :] = shape_vector

    return shape_vectors

class ShapeVectorStorm(StormObject):
    shape_vectors: list[ShapeVector]

    def __init__(
            self, polygon: Polygon, dbz_map: np.ndarray, img_shape: tuple[int, int], id: str = "",
            density: float = DENSITY, radii: list[float] = RADII, num_sectors: int = NUM_SECTORS
        ):
        # initialize with the contour and the id
        super().__init__(contour=polygon, id=id)
        self.img_shape = img_shape
        
        contour = convert_polygons_to_contours([self.contour])[0]
        particles = self._sample_particles(contour, density)

        # create the shape vectors
        vectors = _construct_shape_vector_dbz(
            dbz_map=dbz_map, particles=particles, num_sectors=num_sectors, radii=radii, shape=img_shape
        )

        self.shape_vectors = [ShapeVector(
            coord=(coord[0], coord[1]), vector=vector
        ) for coord, vector in zip(particles.reshape(-1, 2), vectors)]
    
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
import numpy as np
import cv2
from copy import deepcopy
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
import rasterio.features as rfeat
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
from datetime import datetime

from src.cores.base import StormObject
from src.cores.polar_description_vector import ShapeVector, construct_sector
from src.preprocessing import convert_polygons_to_contours
from src.cores.base import StormsMap

from .default import DEFAULT_DENSITY, DEFAULT_NUM_SECTORS, DEFAULT_RADII, DEFAULT_DBZ_THRESHOLD

def scale_map(dbz_map: np.ndarray) -> np.ndarray:
    """
    Scale image to binary map based on threshold.
    1 if dbz >= THRESHOLD else 0
    """
    return (dbz_map >= DEFAULT_DBZ_THRESHOLD).astype(np.float32)

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

def construct_shape_vector_dbz(
        dbz_map: np.ndarray, particles: np.ndarray,
        num_sectors: int = DEFAULT_NUM_SECTORS, radii: list = DEFAULT_RADII, show_progess: bool = True, desc: str = ""
    ) -> np.ndarray:
    """
    Construct shape vectors for given particles based on dbz map.
    """
    # scale the dbz map to (0,1)
    scaled_map = scale_map(dbz_map)
    
    # prepare sector masks
    origin = (radii[-1], radii[-1])

    ## create base sectors
    sector_mask_template = np.zeros(dbz_map.shape, dtype=np.int32)
    prev_base_sector = None

    for r_idx, radius in enumerate(radii):
        new_sector = construct_sector(origin, radius, 0, 360/num_sectors)
        base_sector = new_sector if r_idx == 0 else new_sector.difference(prev_base_sector)
        prev_base_sector = new_sector
        for s_idx in range(num_sectors):
            part_idx = r_idx * num_sectors + s_idx + 1
            sector = rotate(base_sector, s_idx * 360 / num_sectors, origin=origin)

            mask = rfeat.rasterize([(sector, 1)], out_shape=dbz_map.shape, dtype=np.uint8)

            # Direct overwrite
            sector_mask_template[mask > 0] = part_idx

    shape_vectors = np.zeros((particles.shape[0], len(radii) * num_sectors), dtype=np.float32)

    if show_progess:
        pbar = tqdm(enumerate(particles), total=particles.shape[0], leave=False, desc=desc)
    else:
        pbar = enumerate(particles)

    for p_idx, (yoff, xoff) in pbar:    # since particles are in (y, x) order
        sector_mask = shift_mask(sector_mask_template, xoff, yoff, origin)
        sector_values = scaled_map * sector_mask

        shape_vectors[p_idx, :] = np.array([np.sum(sector_values == (i + 1)) for i in range(len(radii) * num_sectors)])

    return shape_vectors

class ShapeVectorStorm(StormObject):
    shape_vectors: list[ShapeVector]
    centroid: tuple[float, float]

    def __init__(
            self, polygon: Polygon, dbz_map: np.ndarray, id: str = "",
            density: float = DEFAULT_DENSITY, radii: list[float] = DEFAULT_RADII, num_sectors: int = DEFAULT_NUM_SECTORS
        ):
        # initialize with the contour and the id
        super().__init__(contour=polygon, id=id)
        contour = convert_polygons_to_contours([self.contour])[0]
        particles = self._sample_particles(contour, density, dbz_map.shape)

        # create the shape vectors
        vectors = construct_shape_vector_dbz(
            dbz_map=dbz_map, particles=particles, num_sectors=num_sectors, radii=radii, 
            desc=f"Constructing shape vectors for {self.id}"
        )

        self.centroid = (self.contour.centroid.y, self.contour.centroid.x)

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
        dy, dx = displacement
        # update contour
        self.contour = translate(self.contour, xoff=dx, yoff=dy)

        # update shape vectors
        for vector in self.shape_vectors:
            vector.coord = vector.coord[0] + dy, vector.coord[1] + dx
        
        # update centroid
        self.centroid = (self.centroid[0] + dy, self.centroid[1] + dx)
    
    def get_num_particles(self):
        """
        Get the number of particles.
        """
        return len(self.shape_vectors)
    
    def _sample_particles(self, contour: np.ndarray, density: float, shape: tuple) -> np.ndarray:
        """
        Sample a list of particles inside the contour.

        Args:
            contour (np.ndarray): the list of points, in order, creating the contour.
            div (float, default): determine the number of particles = area / div.
        
        Returns:
            particles (np.ndarray): the list of particles.
        """
        # get the set of points bounded by the contour
        mask = np.zeros(shape=shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], color=1)
        points = np.argwhere(mask > 0)

        # cluster them 
        n_clusters = int(cv2.contourArea(contour) * density) + 1
        k_means = KMeans(n_clusters, random_state=2025)
        k_means.fit(points)

        return k_means.cluster_centers_.astype(np.int64)    # in (y, x) order

class DbzStormsMap(StormsMap):
    storms: list[ShapeVectorStorm]
    dbz_map: np.ndarray

    def __init__(self, storms: list[ShapeVectorStorm], time_frame: datetime, dbz_map: np.ndarray):
        """
        Beside 2 default attributes, also keep track of `dbz_map` for computing correlation.
        """
        super().__init__(storms, time_frame)
        self.dbz_map = dbz_map
    
    def _retrieve_block_movement(self, block: np.ndarray, search_region: np.ndarray) -> np.ndarray:
        block = block.astype(np.float32)
        search_region = search_region.astype(np.float32)
        result = cv2.matchTemplate(search_region, block, cv2.TM_CCOEFF_NORMED)
        return np.unravel_index(np.argmax(result), result.shape)
    
    def estimate_motion_vector_backtrack(
            self, prev_storms_map: "DbzStormsMap", 
            particles_estimated_movement: list[np.ndarray], **kwargs
        ) -> list[np.ndarray]:
        """
        Estimate motion vector for each storm based on estimated movement from matched particles.

        Args:
        --------
            prev_storms_map (DbzStormsMap): the previous storms map.
            particles_estimated_movement (list[np.ndarray]): list of estimated movement from matched particles.
        
        Returns:
        --------
            motion_vectors (list[np.ndarray]): list of estimated motion vectors for each storm.
        """
        H, W = self.dbz_map.shape
        max_velocity = kwargs.get("max_velocity", 100)
        dt = (self.time_frame - prev_storms_map.time_frame).total_seconds() / 3600.0    # in hr
        buffer = int(max_velocity * dt / 2)         # buffer = half of max_velocity * dt (in pixels)

        motion_vectors = []

        for storm, estimed_movement in zip(self.storms, particles_estimated_movement):
            min_x, min_y, max_x, max_y = storm.contour.bounds

            if estimed_movement is None:
                motion_vectors.append(np.array([None, None], dtype=np.float32))
                continue
                # dy, dx = 0.0, 0.0
            else:
                dy, dx = estimed_movement

            # find search region by shifting the bounding box (-dx, -dy) and adding buffer
            smin_x = max(0, int(min_x - dx - buffer))
            smax_x = min(W, int(max_x - dx + buffer))
            smin_y = max(0, int(min_y - dy - buffer))
            smax_y = min(H, int(max_y - dy + buffer))

            search_region = prev_storms_map.dbz_map[smin_y:smax_y, smin_x:smax_x]
            block = self.dbz_map[int(min_y):int(max_y), int(min_x):int(max_x)]

            block_dy, block_dx = self._retrieve_block_movement(block, search_region)
            trec_dy = min_y - (smin_y + block_dy)
            trec_dx = min_x - (smin_x + block_dx)

            # combine coarse and fine movements
            fine_dy = (trec_dy + dy) / 2.0
            fine_dx = (trec_dx + dx) / 2.0

            motion_vectors.append(np.array([fine_dy, fine_dx], dtype=np.float32) / dt)   # in pixels per hour
        
        return motion_vectors
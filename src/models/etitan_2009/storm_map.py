from datetime import datetime
import numpy as np
import cv2

from src.cores.base import StormsMap
from src.identification import MorphContourIdentifier
from src.preprocessing import convert_contours_to_polygons, convert_polygons_to_contours

from .storm import CentroidStorm

class DbzStormsMap(StormsMap):
    storms: list[CentroidStorm]
    dbz_map: np.ndarray

    def __init__(self, storms: list[CentroidStorm], time_frame: datetime, dbz_map: np.ndarray):
        """
        Beside 2 default attributes, also keep track of `dbz_map` for computin correlation.
        """
        super().__init__(storms, time_frame)
        self.dbz_map = dbz_map

    def _retrieve_movement(self, block: np.ndarray, search_region: np.ndarray) -> np.ndarray:
        block = block.astype(np.float32)
        search_region = search_region.astype(np.float32)
        result = cv2.matchTemplate(search_region, block, cv2.TM_CCOEFF_NORMED)
        return np.unravel_index(np.argmax(result), result.shape)

    
    def trec_estimate(self, other: "DbzStormsMap", **kargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a correlation map which show the movement between the current to other storm map.
        """
        block_size = kargs.get("block_size", 16)
        stride = kargs.get("stride", 16)
        local_buffer = kargs.get("buffer", 50)      # search region = block + expanded by local_buffer

        dbz_map_1 = self.dbz_map
        dbz_map_2 = other.dbz_map
        H, W = dbz_map_2.shape

        ys = list(range(0, H-block_size+1, stride))     # ys: list[start_idx of H-axis]
        xs = list(range(0, W-block_size+1, stride))     # xs: list[start_idx of W-axis]

        vy = np.zeros(shape=(len(ys), len(xs)))         # vy: keep the y-value of movement at corresponding position
        vx = np.zeros_like(vy)                          # vx: keep the y-value of movement at corresponding position

        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                block = dbz_map_1[y:y+block_size, x:x+block_size]
                if np.std(block) < 1e-3:    # case std is too small => continue
                    continue

                # otherwise: get the search region
                y_search_low, y_search_high = max(0,y-local_buffer), min(H,y + block_size + local_buffer)   # ensure the seach region is not overflow.
                x_search_low, x_search_high = max(0,x-local_buffer), min(W,x + block_size + local_buffer)

                search_region = dbz_map_2[y_search_low:y_search_high, x_search_low:x_search_high]
                dy, dx = self._retrieve_movement(block, search_region)

                y_best, x_best = y_search_low + dy, x_search_low + dx
                vy[i][j] = y_best - y
                vx[i][j] = x_best - x
        
        # Get the center of the block
        grid_y = np.array(ys) + block_size / 2
        grid_x = np.array(xs) + block_size / 2

        for storm in self.storms:
            storm.retrieve_movement(grid_y=grid_y, grid_x=grid_x, vy=vy, vx=vx)

        return grid_y, grid_x, vy, vx
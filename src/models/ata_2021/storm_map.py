import numpy as np
from datetime import datetime
from skimage.registration import phase_cross_correlation

from .storm import CentroidStorm
from src.cores.base import StormsMap

class DbzStormsMap(StormsMap):
    storms: list[CentroidStorm]
    dbz_map: np.ndarray

    def __init__(self, storms: list[CentroidStorm], time_frame: datetime, dbz_map: np.ndarray):
        """
        Beside 2 default attributes, also keep track of `dbz_map` for computin correlation.
        """
        super().__init__(storms, time_frame)
        self.dbz_map = dbz_map
    
    def fft_estimate(self, other: "DbzStormsMap", max_velocity: float):
        """
        Estimate the movement for each storm in the current map.
        Return the square, surrounding regions without buffer for each storm.
        """
        H, W = self.dbz_map.shape
        dt = (other.time_frame - self.time_frame).total_seconds() / 3600.0  # in hours
        max_displacement = max_velocity * dt
        buffer = int(max_displacement) + 5          # compute the buffer

        region_list = []
    
        for storm in self.storms:
            min_x, min_y, max_x, max_y = storm.contour.bounds
            y_len = max_y - min_y
            x_len = max_x - min_x
            if y_len < x_len:
                min_y = max(min_y - ((x_len - y_len) // 2), 0)
                max_y = min(max_y + ((x_len - y_len) // 2), H)
                min_x = max(min_x, 0)
                max_x = min(max_x, W)
            else:
                min_x = max(min_x - ((y_len - x_len) // 2), 0)
                max_x = min(max_x + ((y_len - x_len) // 2), W)
                min_y = max(min_y, 0)
                max_y = min(max_y, H)

            # store the region without buffer
            region_list.append((min_y, max_y, min_x, max_x))

            # add buffer to the search region
            min_x = max(min_x - buffer, 0)
            max_x = min(max_x + buffer, W)
            min_y = max(min_y - buffer, 0)
            max_y = min(max_y + buffer, H)

            # get the search regions
            prev_region = self.dbz_map[int(min_y):int(max_y), int(min_x):int(max_x)]
            curr_region = other.dbz_map[int(min_y):int(max_y), int(min_x):int(max_x)]

            # normalize regions
            prev_region = prev_region - np.mean(prev_region)
            curr_region = curr_region - np.mean(curr_region)

            # compute shift via FFT
            shift, error, diffphase = phase_cross_correlation(curr_region, prev_region, upsample_factor=10)
            dy, dx = shift

            # truncate the large shift
            if np.linalg.norm(shift) > max_displacement:
                scale_factor = max_displacement / np.linalg.norm(shift)
                dy *= scale_factor
                dx *= scale_factor

            storm.estimated_velocity = (dy / dt, dx / dt)  # in pixels / hour

        return region_list
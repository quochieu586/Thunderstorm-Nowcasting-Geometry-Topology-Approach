from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter

from src.cores.base import StormsMap

import numpy as np

class FFTMovement:
    max_velocity: float

    def __init__(self, max_velocity: float = 100, smooth_sigma: float = 1.5):
        super().__init__()
        self.max_velocity = max_velocity
        self.smooth_sigma = smooth_sigma

    def estimate_movement(self, prev_map: StormsMap, curr_map: StormsMap):
        H, W = prev_map.dbz_map.shape
        dt = (curr_map.time_frame - prev_map.time_frame).total_seconds() / 3600.0
        if dt <= 0:
            raise ValueError("Non-positive time difference between frames.")

        max_displacement = self.max_velocity * dt
        buffer = int(max_displacement)

        movement_list = []
        region_list = []

        for storm in prev_map.storms:
            min_x, min_y, max_x, max_y = storm.contour.bounds

            # ---- square the window ----
            y_len = max_y - min_y
            x_len = max_x - min_x
            if y_len < x_len:
                pad = (x_len - y_len) // 2
                min_y -= pad
                max_y += pad
            else:
                pad = (y_len - x_len) // 2
                min_x -= pad
                max_x += pad

            # ---- apply buffer ----
            min_x = int(max(min_x - buffer, 0))
            max_x = int(min(max_x + buffer, W))
            min_y = int(max(min_y - buffer, 0))
            max_y = int(min(max_y + buffer, H))

            prev_region = prev_map.dbz_map[min_y:max_y, min_x:max_x]
            curr_region = curr_map.dbz_map[min_y:max_y, min_x:max_x]

            # ---- normalize (mean removal as in paper) ----
            prev_region = prev_region - np.mean(prev_region)
            curr_region = curr_region - np.mean(curr_region)

            # ---- FFT cross-covariance (Leese et al.) ----
            F1 = fft2(prev_region)
            F2 = fft2(curr_region)

            C = np.conj(F1) * F2
            eps = 1e-8
            Cov = np.real(ifft2(C / (np.abs(C) + eps)))
            Cov = fftshift(Cov)

            # ---- Gaussian smoothing (paper step 4) ----
            Cov_smooth = gaussian_filter(Cov, sigma=self.smooth_sigma)

            # ---- locate peak ----
            peak_y, peak_x = np.unravel_index(
                np.argmax(Cov_smooth), Cov_smooth.shape
            )

            center_y = Cov_smooth.shape[0] // 2
            center_x = Cov_smooth.shape[1] // 2

            dy = peak_y - center_y
            dx = peak_x - center_x

            # ---- truncate displacement ----
            disp = np.array([dy, dx], dtype=float)
            norm = np.linalg.norm(disp)
            if norm > max_displacement:
                disp *= max_displacement / norm

            velocity = disp / dt  # pixels per hour

            movement_list.append(velocity)
            region_list.append((min_y, max_y, min_x, max_x))

        return movement_list, region_list
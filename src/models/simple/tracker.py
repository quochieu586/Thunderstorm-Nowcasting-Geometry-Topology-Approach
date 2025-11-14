import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
from scipy.signal.windows import hann

from src.cores.base import StormObject
from src.preprocessing import convert_polygons_to_contours

class PhaseCorrelationTracking:
    def __init__(self, upsample_factor: int=4):
        self.upsample_factor = upsample_factor
        
    def phase_corr_shift(self, prev_storms: list[StormObject], curr_storms: list[StormObject], upsample_factor: int=4) -> tuple[float, float]:
        """
        Compute the phase correlation shift between two storm objects using fast Fourier transform.

        Returns:
            dx (float): Shift in x direction from 
            dy (float): Shift in y direction.
        """
        max_width = int(max(prev_storms[0].contour.bounds[2], curr_storms[0].contour.bounds[2]))
        max_height = int(max(prev_storms[0].contour.bounds[3], curr_storms[0].contour.bounds[3]))

        block1 = np.zeros(shape=(max_height, max_width), dtype=np.float32)
        block2 = np.zeros(shape=(max_height, max_width), dtype=np.float32)

        cv2.drawContours(block1, convert_polygons_to_contours([storm.contour for storm in prev_storms]), -1, (1,), thickness=-1)
        cv2.drawContours(block2, convert_polygons_to_contours([storm.contour for storm in curr_storms]), -1, (1,), thickness=-1)

        # Crop to the bounding box of the combined storms to reduce computation
        min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
        for storm in prev_storms + curr_storms:
            x, y = storm.contour.exterior.xy
            min_x = min(min_x, np.min(x))
            max_x = max(max_x, np.max(x))
            min_y = min(min_y, np.min(y))
            max_y = max(max_y, np.max(y))
        block1 = block1[int(min_y):int(max_y)+1, int(min_x):int(max_x)+1]
        block2 = block2[int(min_y):int(max_y)+1, int(min_x):int(max_x)+1]

        # Apply window to reduce edge effects
        EPS = 1e-9  # Small constant to avoid division by zero
        win = np.outer(hann(block1.shape[0]), hann(block1.shape[1]))
        a = (block1 - np.mean(block1)) * win
        b = (block2 - np.mean(block2)) * win

        F1 = fft2(a)
        F2 = fft2(b)
        R = F1 * np.conj(F2)
        R /= (np.abs(R) + EPS)

        cross = fftshift(ifft2(R).real)  # cross-correlation surface (shifted)
        ny, nx = cross.shape
        # find peak
        peak_idx = np.unravel_index(np.argmax(cross), cross.shape)
        # convert peak location to integer shift (center reference)
        cy, cx = ny//2, nx//2
        int_dy = peak_idx[0] - cy
        int_dx = peak_idx[1] - cx

        if upsample_factor > 1:
            # local upsample around the peak using zero-padding trick
            # Build small window around peak
            win_sz = 8  # small neighborhood for refinement
            y0 = (peak_idx[0] - win_sz//2) % ny
            x0 = (peak_idx[1] - win_sz//2) % nx
            # extract neighborhood (wrap if needed)
            neigh = np.zeros((win_sz, win_sz))
            for i in range(win_sz):
                for j in range(win_sz):
                    neigh[i, j] = cross[(y0 + i) % ny, (x0 + j) % nx]
            # upsample via Fourier zero-pad: take FFT of neigh and pad
            Fn = fft2(neigh)
            pad_y = win_sz * upsample_factor
            pad_x = win_sz * upsample_factor
            big = np.zeros((pad_y, pad_x), dtype=complex)
            cy_n, cx_n = win_sz//2, win_sz//2
            # place centered
            big[:win_sz, :win_sz] = Fn  # small embedding — good for modest upsample
            ups = np.abs(ifft2(big))
            # find refined peak
            pk = np.unravel_index(np.argmax(ups), ups.shape)
            # offset within upsample region
            off_y = (pk[0] - pad_y//2) / upsample_factor
            off_x = (pk[1] - pad_x//2) / upsample_factor
            dy = int_dy + off_y
            dx = int_dx + off_x
        else:
            dy, dx = int_dy, int_dx

        return -dx, -dy
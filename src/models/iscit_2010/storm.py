import numpy as np
import cv2
from sklearn.cluster import KMeans
from shapely.geometry import Polygon

from src.cores.base import StormObject
from src.preprocessing import convert_polygons_to_contours

class ParticleStorm(StormObject):
    particles: list[np.ndarray]  # list of (y,x) coordinates representing particles

    def __init__(
            self, contour: Polygon, history_movements = [], centroid = None, 
            id = "", density: float = 0.05, shape: np.ndarray = None):
        super().__init__(contour, history_movements, centroid, id)
        self._sample_particles(density, shape)
    
    def _sample_particles(self, density: float, shape: np.ndarray) -> None:
        """
        Sample particles within the storm contour.

        Args:
            density (float): Density of particles to sample.
            shape (np.ndarray): Shape of the image containing the storm.
        """
        contour = convert_polygons_to_contours([self.contour])[0]
        # get the set of points bounded by the contour
        mask = np.zeros(shape=shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], color=1)
        points = np.argwhere(mask > 0)

        # uniformly sample points using k-means.
        num_particles = int(cv2.contourArea(contour) * density) + 1
        k_means = KMeans(n_clusters=num_particles)
        k_means.fit(points)

        # particles: list of (y, x)
        self.particles = k_means.cluster_centers_.astype(np.int64)

    def make_move(self, movement):
        super().make_move(movement)
        # update particles
        dy, dx = movement
        for i in range(len(self.particles)):
            self.particles[i][0] += dy
            self.particles[i][1] += dx
    
    def forecast_particles(self, grid_y, grid_x, vy, vx) -> np.ndarray:
        """
        Forecast the positions of particles based on the movement field. 
        Idea: for each particle, retrieve its nearest movement vector and update its position accordingly.
        """
        forecasted_particles = self.particles.copy()
        for i in range(len(forecasted_particles)):
            y, x = forecasted_particles[i]
            # get movement vector
            grid_y_idx = np.clip(np.searchsorted(grid_y, y) - 1, 0, len(grid_y)-2)
            grid_x_idx = np.clip(np.searchsorted(grid_x, x) - 1, 0, len(grid_x)-2)
            dy = vy[grid_y_idx, grid_x_idx]
            dx = vx[grid_y_idx, grid_x_idx]
            # update particle position
            forecasted_particles[i][0] += dy
            forecasted_particles[i][1] += dx
        
        return forecasted_particles
    
    def get_num_particles(self):
        """
        Get the number of particles.
        """
        return len(self.particles)
    
    def count_particles_in_storm(self, particles: np.ndarray, img_shape: list) -> int:
        """
        Count the number of particles inside the given contour.
        """
        mask = np.zeros(shape=img_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, convert_polygons_to_contours([self.contour]), color=1)

        y_points = particles[:, 0]
        x_points = particles[:, 1]

        particle_mask = np.zeros(shape=img_shape[:2], dtype=np.uint8)
        particle_mask[y_points, x_points] = 1

        return np.sum(mask * particle_mask)

import ripser
from persim import plot_diagrams
import numpy as np
from scipy.spatial import distance_matrix

import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from utils.preprocessing import extract_contour_by_dbz
from app.src.config import LEGEND_DIR

class PersistenceDiagram:
    @staticmethod
    def _get_threshold_distance(X: np.ndarray) -> float:
        """
            Compute the minimum of maximum distance between point X and other points.
            Use for determine threshold distance in Vietoris-Rips complex to be reduced time complexity.
        """
        dist_matrix = distance_matrix(X, X)
        return np.min(np.max(dist_matrix, axis=0))
    
    @staticmethod
    def compute(data_cloud: np.ndarray, maxdim: int = 1) -> dict:
        """
            Compute persistence diagram using Vietoris-Rips complex.
            Args:
                X: np.ndarray of shape (n_points, n_dimensions)
                maxdim: maximum homology dimension to be computed
                thresh: threshold distance to be used in Vietoris-Rips complex
        """
        return ripser.ripser(X=data_cloud, maxdim=maxdim, thresh=PersistenceDiagram._get_threshold_distance(data_cloud))
    
    @staticmethod
    def plot_persistence_diagram(dsgm: list[np.ndarray], title: str = 'Persistence Diagram') -> None:
        """
            Plot persistence diagram.
            Args:
                dgms: list of np.ndarray, each np.ndarray is a persistence diagram of a homology dimension
                homology_dims: list of homology dimensions to be plotted
                max_thresh: maximum threshold distance to be used in plot
        """
        plot_diagrams(dsgm, show=True, title=f'Persistence Diagram')
        return
    
    @staticmethod
    def compute_and_plot_persistence_homology(image_path: str, threshold: int = 20, contour_threshold: int = 100):
        # Load color legend
        with open(os.path.join(LEGEND_DIR, 'color_dbz.json')) as f:
            list_color = json.load(f)
        sorted_color = sorted({tuple(color[1]): color[0] for color in list_color}.items(), key=lambda item: item[1])

        source_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if source_img is None:
            raise FileNotFoundError(f"Failed to load {image_path}")
        
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

        _, contours, _  = extract_contour_by_dbz(source_img, thresholds=[threshold], sorted_color=sorted_color)
        contours = contours[0]
        
        # Remove small contours
        contours = sorted([polygon for polygon in contours if cv2.contourArea(polygon) > contour_threshold], key=lambda x: cv2.contourArea(x), reverse=True)

        if not contours:
            return None, None

        data_cloud = np.concatenate([contour.squeeze() for contour in contours])
        res_pc = PersistenceDiagram.compute(data_cloud, maxdim=1)

        # Plot original image with contours
        img_with_contours = source_img.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 0, 0), 2)

        # Create a new figure for the persistence diagram
        fig = plt.figure()
        PersistenceDiagram.plot_persistence_diagram(res_pc['dgms'], title=f'Persistence Diagram for {os.path.basename(image_path)} at threshold {threshold}')

        return img_with_contours, fig
        
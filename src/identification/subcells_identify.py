import numpy as np
import cv2
from typing import List, Tuple
import numpy as np
from skimage.graph import MCP

from src.preprocessing import _preprocess, _convert_to_dbz
from .base import BaseStormIdentifier

from src.preprocessing import sorted_color

class SubcellStormIdentifier(BaseStormIdentifier):
    """
        Detect storm objects solely based on the contiguous spatial areas of pixels exceeding specified dBZ thresholds. 
    """
    def identify_storm(
        self,
        img: np.ndarray, 
        threshold: int, 
        filter_area = 15,
        distance_dbz_threshold = 8,
        subcells_check = True
    ):
        """
            Draw the DBZ contour for the image.

            Args:
                img: source image.
                thresholds: dbz thresholds for drawing contours.
                sorted_color: a list of tuples where each tuple includes:
                    - a color represented as an RGB triplet (e.g., (R, G, B)).
                    - a corresponding dBZ value, sorted in increasing order of dBZ.
            
            Returns:
                Tuple[np.ndarray,List[np.ndarray]]: A tuple containing:
                    - contour_img (np.ndarray): A blank image with the extracted contours drawn.
                    - contours (List[np.ndarray]): A list of detected contours, each represented as an array of points.
                    - contour_colors (List(Tuple[int,int,int])): A list of color corresponding with each contours
        """
        img = _preprocess(img)

        dbz_map = _convert_to_dbz(img, sorted_color).astype(np.uint8)
        contours_time = []
        region = (dbz_map >= threshold).astype(np.uint8)

        # Draw the contour
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        if subcells_check:
            processed_contours = []
            for contour in contours:
                if cv2.contourArea(contour) < filter_area:
                    continue
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [contour], color=1)

                M = float(np.where(mask, dbz_map, 0).max())
                F = float(threshold)
                D = float(distance_dbz_threshold)

                processed_contours.extend(self._process_subcells(dbz_map, mask, F, M, D))

            contours_time.append(processed_contours)
        else:    
            contours_time = contours

        return dbz_map, contours_time

    def _process_subcells(self, dbz_map: np.ndarray, mask: np.ndarray, F: float, M: float, D:float):
        """
            Determine subcells of a cell.

            Args:
                dbz_map (np.ndarray): DBz map.
                mask (np.ndarray): a binary mask indicating the storm pixels.
                F (float): the minimum field value. 
                M (float): the maximum field value.
                D (float): the distance between each jump.
        """
        n = 1
        while True:
            try:
                Hn = M - n * D
            except Exception as e:
                print(f"Hn = {Hn}\n{e}")

            if Hn <= D + F:         # Case assumption 1 is broken.
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                return contours

            submask = ((dbz_map >= Hn) & (mask > 0)).astype(np.uint8)
            kernel = np.ones((3,3), dtype=np.uint8)
            dilate_submask = (cv2.dilate(submask, kernel) & (mask > 0))

            num_labels, labels = cv2.connectedComponents(dilate_submask, connectivity=8)

            if num_labels > 2:     # Case assumption 2 is broken: new subcells created.
                contours = []
                nearest = self._assign_subcells(mask, labels)

                for l in range(1, num_labels):
                    submask = np.where(nearest == l, 1, 0).astype(np.uint8)
                    contours.extend(self._process_subcells(dbz_map, submask, F=F, M=Hn, D=D))
                
                return contours

            n += 1

    def _assign_subcells(mask, subcell_labels):
        costs = np.where(mask, 1.0, np.inf)  # allow paths only inside mask
        mcp = MCP(costs)

        sources = np.transpose(mask.nonzero())
        destinations = np.transpose(subcell_labels.nonzero())

        mcp.find_costs(starts=destinations)
        nearest = mask.copy()

        for src in sources:
            s, r = src
            dest = mcp.traceback(src)[0]
            nearest[s][r] = subcell_labels[dest]

        return nearest
from .base import BaseStormIdentifier
import cv2
import numpy as np
from skimage.graph import MCP

class HypothesisIdentifier(BaseStormIdentifier):
    """
    Implementation of storm identification using hypothesis testing in paper *The Australian Identification, Nowcasting and Tracking Algorithm (A.I.N.T.)*.

    Args:
        dbz_map (np.ndarray): a 2D array of reflectivity values.
        threshold (int): dbz threshold for drawing contours.
        filter_area (float): minimum area of storm to be considered.
        distance_dbz_threshold (float): the distance between each jump.
        filter_center (float): minimum area of substorm center to be considered.
    
    Returns:
        contours (list[np.ndarray]): List of contours.
    """

    def __init__(self, threshold: int, filter_area: int = 20, distance_dbz_threshold: int = 5, filter_center: float = 10):
        self.threshold = threshold
        self.filter_area = filter_area
        self.distance_dbz_threshold = distance_dbz_threshold
        self.filter_center = filter_center

    def identify_storm(
        self, dbz_map: np.ndarray
    ) -> list[np.ndarray]:
        """
            Draw the DBZ contour for the image.

            Args:
                dbz_map: image where each pixel represents the dBZ value.
            Returns:
                List[np.ndarray]: A list of detected contours, each represented as an array of points.
        """
        contours = []
        lowest_mask = (dbz_map >= self.threshold).astype(np.uint8)

        # extract connected components
        num_labels, labels = cv2.connectedComponents(lowest_mask, connectivity=8)
        
        for label in range(1, num_labels):
            roi_mask = (labels == label).astype(np.uint8)
            if np.sum(roi_mask) < self.filter_area:  # case small storm => filter out
                continue

            M = float(np.where(roi_mask, dbz_map, 0).max())
            F = float(self.threshold)
            D = float(self.distance_dbz_threshold)

            contours.extend(self._process_subcells(dbz_map, roi_mask, F, M, D, self.filter_center))

        return contours

    def _process_subcells(self, dbz_map: np.ndarray, mask: np.ndarray, F: float, M: float, D:float, filter_center:float) -> np.ndarray:
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

            if Hn <= D + F:             # Case assumption 1 is broken.
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                return contours

            submask = ((dbz_map >= Hn) & (mask > 0)).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            dilate_submask = (cv2.dilate(submask, kernel) & (mask > 0))

            num_labels, labels = cv2.connectedComponents(dilate_submask, connectivity=8)

            valid_mask = np.zeros_like(labels, dtype=np.uint8)
            num_valid = 0

            # only assign valid mask for those labels with num > filter_center
            for label in range(1, num_labels):
                if np.sum(labels == label) >= filter_center:
                    valid_mask = np.sum([valid_mask, (labels == label).astype(np.uint8)], axis=0)
                    num_valid += 1
            
            valid_mask = valid_mask.astype(bool)
            valid_labels = np.where(valid_mask, labels, 0)

            if num_valid > 1:     # Case assumption 2 is broken: new subcells created.
                contours = []
                nearest = self._assign_subcells(mask, valid_labels)

                for l in range(1, num_labels):
                    submask = np.where(nearest == l, 1, 0).astype(np.uint8)
                    if np.sum(submask) == 0:
                        continue
                    contours.extend(self._process_subcells(dbz_map, submask, F=F, M=Hn, D=D, filter_center=filter_center))
                
                return contours

            n += 1

    def _assign_subcells(self, mask, subcell_labels):
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
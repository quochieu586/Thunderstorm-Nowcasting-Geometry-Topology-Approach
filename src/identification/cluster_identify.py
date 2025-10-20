from .base import BaseStormIdentifier
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

DISTANCE = 1.1
MIN_SAMPLES = 3

class ClusterIdentifier(BaseStormIdentifier):
    def __init__(self, thresholds: list[float], filter_area: float, filter_center: float):
        self.thresholds = thresholds
        self.filter_area = filter_area
        self.filter_center = filter_center

    def set_params(self, threshold: int, filter_area: float):
        self.filter_area = filter_area
        self.thresholds = [t for t in range(threshold, threshold+25, 5)]

    def identify_storm(self, dbz_map: np.ndarray) -> list[np.ndarray]:
        """
        Implementation of storm identification using clustering algorithm (DBSCAN) in paper *An Improved Storm Cell Identification and Tracking (SCIT) Algorithm based on DBSCAN Clustering and JPDA Tracking Methods*.

        Args:
            dbz_map (np.ndarray): a 2D array of reflectivity values.
        
        Returns:
            contours (list[np.ndarray]): List of contours.
        """
        substorms_list = [self._extract_substorms(dbz_map, threshold, self.filter_area) for threshold in self.thresholds]
        storms = [storm for storm in substorms_list[0] if np.sum(storm) >= self.filter_area]

        for substorms in substorms_list[1:]:
            if len(substorms) == 0:
                break
            storms = self._process_storms(storms, substorms, filter_center=self.filter_center)

        # return storms
        contours = []
        for storm in storms:
            contours.extend(cv2.findContours(storm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        
        return contours


    def _extract_substorms(
        self, dbz_map: np.ndarray, threshold: float, filter_area: float,
        eps: float = DISTANCE, min_samples: int = MIN_SAMPLES,
    ):
        mask = dbz_map > threshold
        points = np.argwhere(mask)

        if len(points) < filter_area:
            return []
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_ + 1         # add 1 to resolve -1 which means noisy point.

        label_map = np.zeros_like(mask, dtype=np.uint16)
        x, y = points[:, 0], points[:, 1]
        label_map[x, y] = labels

        # contours = []
        masks = []
        for label in np.unique(labels):
            if label == 0:
                continue
            color = np.random.randint(0, 255, size=(3,))
            color = tuple(int(c) for c in color)
            roi_mask = (label_map == label).astype(np.uint8)

            masks.append(roi_mask)
        
        return masks

    def _process_storms(self, current_storms: np.ndarray, substorms: np.ndarray, filter_center: float) -> np.ndarray:
        """
        If any storm in current_storms contains more than 1 substorms, then canceling it and append new substorms with expanding to cover full area of the current storm.
        Args:
            current_storms (np.ndarray): list of masks of current storms.
            inside_masks (np.ndarray): list of masks of considered 
        """
        new_storms = []
        for storm in current_storms:
            inside_substorms = []
            outside_substorms = []

            for substorm in substorms:
                if np.sum(substorm) < filter_center:    # small center => filter out
                    continue
                elif np.all(storm >= substorm):         # case substorm is totally inside storm
                    inside_substorms.append(substorm)
                else:
                    outside_substorms.append(substorm)
            
            if len(inside_substorms) <= 1:
                new_storms.append(storm)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                storms = self._competitive_dilation(inside_substorms, storm, kernel)
                new_storms.extend(storms)
                # new_storms.extend(_assign_subcells(storm, inside_substorms))
            
            substorms = outside_substorms
        
        return new_storms

    def _competitive_dilation(self, sub_masks, outer_mask, kernel, max_iter=200):
            """
            Dilate multiple sub-storm masks simultaneously within outer_mask,
            ensuring no overlap between them.
            """
            current_masks = sub_masks.copy()
            combined_mask = np.clip(np.sum(current_masks, axis=0), 0, 1).astype(np.uint8)

            for _ in range(max_iter):
                grown_masks = []
                changed = False

                for mask in current_masks:
                    dilated = cv2.dilate(mask, kernel, iterations=1)
                    # only allow dilation into available space (not occupied by others)
                    new_mask = dilated & (1 - combined_mask) & outer_mask
                    new_mask = np.clip(mask + new_mask, 0, 1)
                    grown_masks.append(new_mask)
                    if np.any(new_mask != mask):
                        combined_mask = np.clip(np.sum([combined_mask, new_mask], axis=0), 0, 1).astype(np.uint8)
                        changed = True

                current_masks = grown_masks

                if not changed:
                    break

            return current_masks
from datetime import datetime
import numpy as np

from src.identification import SimpleContourIdentifier
from src.cores.contours import StormsMap, StormObject
from src.nowcasting import ConstantVectorMotion
from src.cores.metrics import overlapping_area, recall, precision, f1_score

from .base import BasePrecipitationModeling

class ConstantVectorPrecipitationModeling(BasePrecipitationModeling):
    """
    A simple implementation of a precipitation modeling using constant vector approach. Approach for each step including:
    - Identification: identify single storm as contiguous area with dBZ >= THRESHOLD_DBZ.
    - Motion estimation: use single global vector for all storms, estimated by averaging motion of all identified storms.
    - Storm matching: use overlapping area between predicted storm and actual storm to match.
    - Prediction: move each storm according to the global motion vector.
    """
    def __init__(self):
        self.global_motion = ConstantVectorMotion()
        self.identifier = SimpleContourIdentifier()
        self.maps = []

    def _construct_storms_map(self, radar_image: np.ndarray, time_frame: datetime) -> StormsMap:
        contours = self.identifier.identify_storm(radar_image)
        storms = [StormObject(contour=cnt) for cnt in contours]
        return StormsMap(
            global_motion=self.global_motion,
            storms=storms,
            time_frame=time_frame,
            map_size=radar_image.shape
        )
    
    # def _matching(self, predicted_map: StormsMap, true_map: StormsMap) -> list[tuple[StormObject, StormObject]]:
        

    def fit(self, train_data: list[tuple[np.ndarray, datetime]]):
        if len(train_data) < 2:
            raise ValueError("At least two StormsMap objects are required for training.")

        contours_maps = [self._construct_storms_map(img, time) for img, time in train_data]

        current_map = contours_maps[0]
        for next_map in contours_maps[1:]:
            next_storms_map = self._construct_storms_map(next_map[0], next_map[1])




    def predict(self, previous_map: list[StormsMap]) -> StormsMap:
        return self.model.predict(previous_map)
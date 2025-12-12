from datetime import datetime
import numpy as np

from src.preprocessing import convert_contours_to_polygons
from models.base.base import BasePrecipitationModel
from src.cores.base import StormsMap
from src.identification import SimpleContourIdentifier

from ..base.base import BasePrecipitationModel
from .storm import StormShapeVectors
from .tracker import PhaseCorrelationTracking
from .matcher import PolarVectorMatcher

class SimplePrecipitationModel(BasePrecipitationModel):
    """
    Simple precipitation modeling using contour-based storm identification.

    Attributes:
        identifier (SimpleContourIdentifier): The storm identifier used for identifying storms in radar images.
    """
    identifier: SimpleContourIdentifier
    matcher: PolarVectorMatcher
    storms_maps: list[StormsMap]

    def __init__(self, identifier: SimpleContourIdentifier):
        self.identifier = identifier
        self.storms_maps = []
        self.matcher = PolarVectorMatcher(discard_threshold=0.2)
        self.tracker = PhaseCorrelationTracking(upsample_factor=4)

    def identify_storms(self, dbz_img: np.ndarray, time_frame: datetime, map_id: str, threshold: int, filter_area: float) -> StormsMap:
        """
        Identify storms in the given DBZ image.

        Args:
            dbz_img (np.ndarray): The DBZ image.
            time_frame (datetime): The time frame of the image.
            map_id (str): The identifier for the map. Use as prefix for storm IDs.
            threshold (int): The DBZ threshold for storm identification.
            filter_area (float): The minimum area to filter storms.

        Returns:
            StormsMap: The identified storms map.
        """
        contours = convert_contours_to_polygons(self.identifier.identify_storm(dbz_img, threshold=threshold, filter_area=filter_area))
        contours = [contour for contour in contours if contour.area >= filter_area]

        storms_lst = [StormShapeVectors(contour=contour, id=f"{map_id}_storm_{i}") for i, contour in enumerate(contours)]
        [storm.extract_shape_vectors(global_contours=contours) for storm in storms_lst]


        return StormsMap(storms=storms_lst, time_frame=time_frame)

    def processing_map(self, current_map: StormsMap) -> int:
        """
        First match all storms in this map to the previous map, update storms accordingly and append this map into the tracking history.

        Returns:
            int: The number of matched storms from previous map to current map.
        """
        num_matches = 0
        if len(self.storms_maps) >= 1:
            # Match storms between previous and current maps
            previous_map = self.storms_maps[-1]
            matches = self.matcher.match_storms(previous_map, current_map)
            num_matches = len(matches)

            # Estimate shifts and update storm positions
            for prev_idx, curr_idx in matches:
                prev_storm = previous_map.storms[prev_idx]
                curr_storm = current_map.storms[curr_idx]

                dy, dx = self.tracker.phase_corr_shift(
                    prev_storms=[prev_storm],
                    curr_storms=[curr_storm],
                    upsample_factor=self.tracker.upsample_factor
                )
                # print(f"Tracking storm {prev_storm.id} to {curr_storm.id} with shift dx: {dx}, dy: {dy}")
                curr_storm.track_history(prev_storm, optimal_movement=(dx, dy))
        
        self.storms_maps.append(current_map)
        print(f"Total matched storms: {num_matches} over number of storms in current map: {len(current_map.storms)}")
        return num_matches
    
    # def prediction(self, lead_time: int) -> StormsMap:

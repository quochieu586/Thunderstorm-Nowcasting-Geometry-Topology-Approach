from datetime import datetime
import numpy as np

from src.preprocessing import convert_contours_to_polygons
from src.models.base import BasePrecipitationModeling
from src.cores.base import StormsMap
from src.identification import SimpleContourIdentifier

from .storm_with_shape_vectors import StormShapeVectors

class SimpleModel(BasePrecipitationModeling):
    """
    Simple precipitation modeling using contour-based storm identification.

    Attributes:
        identifier (SimpleContourIdentifier): The storm identifier used for identifying storms in radar images.
    """
    identifier: SimpleContourIdentifier
    storms_maps: list[StormsMap]

    def __init__(self, identifier: SimpleContourIdentifier):
        self.identifier = identifier
        self.storms_maps = []

    def identify_storms(self, dbz_img: np.ndarray, timeframe: datetime, map_id: str) -> StormsMap:
        """
        Identify storms in the given DBZ image.

        Args:
            dbz_img (np.ndarray): The DBZ image.
            timeframe (datetime): The time frame of the image.

        Returns:
            StormsMap: The identified storms map.
        """
        contours = convert_contours_to_polygons(self.identifier.identify_storm(dbz_img))

        storms_lst = [
            StormShapeVectors(contour=contour, id=f"{map_id}_storm_{i}") for i, contour in enumerate(contours)
        ]
        [storm.extract_shape_vectors(global_contours=contours) for storm in storms_lst]


        return StormsMap(storms=storms_lst, timeframe=timeframe)

    def processing_map(self, current_map: StormsMap):
        """
        First match all storms in this map to the previous map, update storms accordingly and append this map into the tracking history.
        """
        
        

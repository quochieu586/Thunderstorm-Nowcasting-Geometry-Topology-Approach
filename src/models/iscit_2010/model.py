import cv2
import numpy as np
from datetime import datetime, timedelta

from src.cores.base import StormObject, StormsMap
from src.models.base.model import BasePrecipitationModel
from src.identification import MorphContourIdentifier
from src.preprocessing import convert_contours_to_polygons, convert_polygons_to_contours
from src.cores.movement_estimate import BaseTREC, TREC

from .matcher import STitanMatcher, MatchedStormPair
from ..base.tracker import TrackingHistory, UpdateType
from .storm import ParticleStorm

DENSITY = 0.05  # particle density

class ISCITPrecipitationModel(BasePrecipitationModel):
    """
    ISCIT model implementation for thunderstorm nowcasting.
    """
    identifier: MorphContourIdentifier
    storms_maps: list[StormsMap]
    tracker: TrackingHistory
    matcher: STitanMatcher

    def __init__(self, identifier: MorphContourIdentifier, trec: BaseTREC = None, 
                 max_velocity: float = 100.0, weights: tuple[float, float] = (0.5, 0.5)):
        self.identifier = identifier
        self.storms_maps = []
        if trec is None:
            trec = TREC(max_velocity=max_velocity)

        self.matcher = STitanMatcher(trec=trec, max_velocity=max_velocity, weights=weights)
        self.tracker = None

    def identify_storms(
            self, dbz_map: np.ndarray, time_frame: datetime, map_id: str, threshold: float, filter_area: float, particle_density: float = DENSITY
        ) -> StormsMap:
        """
        Identify storms in the given DBZ image at the specified timestamp.

        Args:
            dbz_img (np.ndarray): The DBZ image.
            time_frame (datetime): The timestamp of the image.
            map_id (str): The identifier for the storm map. Use for prefixing storm IDs.
            threshold (float): The DBZ threshold for storm identification.
            filter_area (float): The minimum area to filter storms.

        Returns:
            StormsMap: The identified storms map.
        """
        polygons = convert_contours_to_polygons(self.identifier.identify_storm(dbz_map=dbz_map, threshold=threshold, filter_area=filter_area))
        polygons = sorted(polygons, key=lambda x: x.area, reverse=True)
        storms = []

        for idx, polygon in enumerate(polygons):
            contour = convert_polygons_to_contours([polygon])[0]

            # Create the mask of current storm
            mask = np.zeros_like(dbz_map, dtype=np.uint8)
            cv2.fillPoly(mask, contour, color=1)

            # Extract DBZ values inside mask
            weights = dbz_map * mask
            y_idx, x_idx = np.indices(dbz_map.shape)
            total_weight = weights.sum()

            if total_weight == 0:
                centroid = (np.nan, np.nan)  # or fallback
            else:
                cx = (x_idx * weights).sum() / total_weight
                cy = (y_idx * weights).sum() / total_weight
                centroid = (int(cy), int(cx))

            storms.append(
                ParticleStorm(polygon, centroid=centroid, id=f"{map_id}_storm_{idx}", shape=dbz_map.shape[:2], density=particle_density)
            )
            
        return StormsMap(storms=storms, time_frame=time_frame, dbz_map=dbz_map)
    
    def processing_map(self, curr_storms_map: StormsMap):
        if len(self.storms_maps) == 0:
            self.storms_maps.append(curr_storms_map)
            self.tracker = TrackingHistory(curr_storms_map)
        else:
            prev_storms_map = self.storms_maps[-1]
            dt = (curr_storms_map.time_frame - prev_storms_map.time_frame).seconds / 3600   # scaled to hour

            # match using Hungarian algorithm
            first_guess = len(self.storms_maps) == 1
            matched: list[MatchedStormPair] = self.matcher.match_storms(
                prev_storms_map, curr_storms_map, first_guess=first_guess
            )

            for info in matched:
                if info.update_type == UpdateType.NEW:
                    self.tracker.add_new_track(
                        new_storm=curr_storms_map.storms[info.curr_storm_order],
                        time_frame=curr_storms_map.time_frame
                    )
                else:
                    self.tracker.update_track(
                        prev_storm=prev_storms_map.storms[info.prev_storm_order],
                        curr_storm=curr_storms_map.storms[info.curr_storm_order],
                        update_type=info.update_type,
                        time_frame=curr_storms_map.time_frame,
                        velocity=info.derive_motion_vector(dt=dt)
                    )

            self.storms_maps.append(curr_storms_map)

    def forecast(self, lead_time: float) -> StormsMap:
        """
        Predict future storms up to lead_time based on the current storm map.

        Args:
            lead_time (float): The lead time in second for prediction.
        """
        dt = lead_time / 3600  # scaled to hour
        current_map = self.storms_maps[-1]
        new_storms = []
        for storm in current_map.storms:
            new_storms.append(storm.forecast(dt))
        
        return StormsMap(storms=new_storms, time_frame=current_map.time_frame + timedelta(hours=dt), dbz_map=None)
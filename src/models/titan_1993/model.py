import numpy as np
from datetime import datetime, timedelta
import cv2
from shapely.geometry import Point, Polygon

from src.cores.base import StormsMap, StormObject
from src.identification import SimpleContourIdentifier
from src.preprocessing import convert_contours_to_polygons, convert_polygons_to_contours

from ..base.model import BasePrecipitationModel
from ..base.tracker import TrackingHistory, UpdateType, MatchedStormPair
from .matcher import SimpleMatcher

class TitanPrecipitationModel(BasePrecipitationModel):
    """
    Precipitation modeling provided in Titan 1993 paper.
    """
    identifier: SimpleContourIdentifier
    storms_maps: list[StormsMap]
    tracker: TrackingHistory
    matcher: SimpleMatcher

    def __init__(self, identifier: SimpleContourIdentifier, max_velocity: float):
        self.identifier = identifier
        self.tracker = None
        self.matcher = SimpleMatcher(max_velocity=max_velocity)
        self.storms_maps = []

    def identify_storms(self, dbz_img: np.ndarray, time_frame: datetime, map_id: str, threshold: float, filter_area: float) -> StormsMap:
        polygons = convert_contours_to_polygons(self.identifier.identify_storm(dbz_img, threshold=threshold, filter_area=filter_area))
        polygons = sorted(polygons, key=lambda x: x.area, reverse=True)
        storms = []

        for idx, polygon in enumerate(polygons):
            contour = convert_polygons_to_contours([polygon])[0]

            # Create the mask of current storm
            mask = np.zeros_like(dbz_img, dtype=np.uint8)
            cv2.fillPoly(mask, contour, color=1)

            # Extract DBZ values inside mask
            weights = dbz_img * mask

            # Create coordinate grids
            y_idx, x_idx = np.indices(dbz_img.shape)

            # Compute weighted centroid
            total_weight = weights.sum()
            if total_weight == 0:
                centroid = (np.nan, np.nan)  # or fallback
            else:
                cx = (x_idx * weights).sum() / total_weight
                cy = (y_idx * weights).sum() / total_weight
                centroid = (int(cy), int(cx))

            storms.append(StormObject(
                polygon, centroid=centroid, id=f"{map_id}_storm_{idx}"
            ))
        
        return StormsMap(storms=storms, time_frame=time_frame, dbz_map=dbz_img)

    def processing_map(self, curr_storms_map: StormsMap) -> int:
        if self.storms_maps == []:
            self.storms_maps.append(curr_storms_map)
            self.tracker = TrackingHistory(curr_storms_map)
        else:
            prev_storms_map = self.storms_maps[-1]
            dt = (curr_storms_map.time_frame - prev_storms_map.time_frame).seconds / 3600   # scaled to hour

            # match using Hungarian algorithm
            matched: list[MatchedStormPair] = self.matcher.match_storms(prev_storms_map, curr_storms_map)

            for info in matched:
                # print(f"Matched: Prev Storm {prev_storms_map.storms[info.prev_storm_order].id} -> Curr Storm {curr_storms_map.storms[info.curr_storm_order].id} | Update Type: {info.update_type.name}")
                try:
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
                except Exception as e:
                    print(f"Error at matching storms: {info.prev_storm_order} -> {info.curr_storm_order} | Update Type: {info.update_type.name}")
                    raise e

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
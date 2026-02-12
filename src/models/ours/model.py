import numpy as np
from torch import from_numpy
from datetime import datetime, timedelta
from tqdm.notebook import tqdm

from src.cores.base import StormsMap
from src.identification import BaseStormIdentifier, HypothesisIdentifier
from src.preprocessing import convert_contours_to_polygons
from src.models.base.model import BasePrecipitationModel
from src.models.base.tracker import TrackingHistory, UpdateType

from src.cores.polar_description_vector import fft_conv2d, construct_polar_kernels

from .matcher import StormMatcher
from .storm import DbzStormsMap, ShapeVectorStorm
from ..base.tracker import MatchedStormPair

from .default import DEFAULT_MAX_VELOCITY, DEFAULT_WEIGHTS, DEFAULT_COARSE_MATCHING_THRESHOLD, DEFAULT_FINE_MATCHING_THRESHOLD, DEFAULT_DENSITY, DEFAULT_NUM_SECTORS, DEFAULT_RADII

class OursPrecipitationModel(BasePrecipitationModel):
    """
    Simple precipitation modeling using contour-based storm identification.

    Attributes:
        identifier (SimpleContourIdentifier): The storm identifier used for identifying storms in radar images.
    """
    identifier: HypothesisIdentifier
    matcher: StormMatcher
    tracker: TrackingHistory
    storms_maps: list[StormsMap]

    def __init__(self, identifier: HypothesisIdentifier, max_velocity: float = DEFAULT_MAX_VELOCITY, weights: tuple[float, float] = DEFAULT_WEIGHTS,
                 radii: list[int] = DEFAULT_RADII, num_sectors: int = DEFAULT_NUM_SECTORS, density: float = DEFAULT_DENSITY):
        self.identifier = identifier
        self.storms_maps = []
        self.matcher = StormMatcher(max_velocity=max_velocity, weights=weights)
        self.tracker = None

        self.radii = radii
        self.num_sectors = num_sectors
        self.density = density

        self.kernels = construct_polar_kernels(radii, num_sectors)

    # def identify_storms(self, dbz_img: np.ndarray, time_frame: datetime, map_id: str, 
    #                     threshold: int, filter_area: float, show_progress: bool = True) -> StormsMap:
    #     contours = self.identifier.identify_storm(dbz_img, threshold=threshold, filter_area=filter_area)
    #     polygons = convert_contours_to_polygons(contours)
    #     polygons = sorted(polygons, key=lambda x: x.area, reverse=True)

    #     pbar = tqdm(enumerate(polygons), total=len(polygons), desc="Constructing ShapeVectorStorms", leave=False) \
    #         if show_progress else enumerate(polygons)

    #     # Construct storms map
    #     storms = [ShapeVectorStorm(
    #                 polygon=polygon, 
    #                 id=f"{map_id}_storm_{idx}",
    #                 dbz_map=dbz_img,
    #                 density=self.density,
    #                 radii=self.radii,
    #                 num_sectors=self.num_sectors
    #             ) for idx, polygon in pbar]
        
    #     return DbzStormsMap(storms, time_frame=time_frame, dbz_map=dbz_img)
    
    def identify_storms(
            self, dbz_img: np.ndarray, time_frame: datetime, map_id: str, threshold: int, filter_area: float
        ) -> DbzStormsMap:
        contours = self.identifier.identify_storm(dbz_img, threshold=threshold, filter_area=filter_area)
        polygons = convert_contours_to_polygons(contours)
        polygons = sorted(polygons, key=lambda x: x.area, reverse=True)

        # Pre-compute the convolution of the dbz map with all sector kernels
        img = from_numpy(dbz_img >= threshold).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        sectors_convolved_np = fft_conv2d(img=img, kernel=from_numpy(self.kernels).unsqueeze(1).float())

        storms = [ShapeVectorStorm(
            polygon=polygon, 
            id=f"{map_id}_storm_{idx}",
            dbz_map=dbz_img,
            density=self.density,
            sectors_convolved_np=sectors_convolved_np,
        ) for idx, polygon in enumerate(polygons)]

        return DbzStormsMap(storms, time_frame=time_frame, dbz_map=dbz_img)

    def processing_map(self, curr_storms_map: StormsMap, coarse_threshold: float = DEFAULT_COARSE_MATCHING_THRESHOLD, fine_threshold: float = DEFAULT_FINE_MATCHING_THRESHOLD) -> int:
        if len(self.storms_maps) == 0:
            self.tracker = TrackingHistory(curr_storms_map)
            update_list: list[MatchedStormPair] = []
        else:
            prev_storms_map = self.storms_maps[-1]
            dt = (curr_storms_map.time_frame - prev_storms_map.time_frame).total_seconds() / 3600  # scaled to hour

            if curr_storms_map.time_frame <= prev_storms_map.time_frame:
                raise ValueError("Current storms map time frame must be later than the previous one.")
            
            update_list = self.matcher.match_storms(
                storms_map_lst_1=prev_storms_map,
                storms_map_lst_2=curr_storms_map,
                coarse_threshold=coarse_threshold,
                fine_threshold=fine_threshold
            )

            for info in update_list:
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
                        velocity=info.derive_motion_vector(dt)
                    )

        self.storms_maps.append(curr_storms_map)
        right_matches = list(set([info.curr_storm_order for info in update_list]))
        return len(right_matches)
    
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
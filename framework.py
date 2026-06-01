from src.models.ours import OursPrecipitationModel
from src.cores.base import StormsMap

from src.identification import MorphContourIdentifier
import numpy as np
import cv2
from datetime import datetime

class SimpleInterfacePrecipitationModel:
    """
    In this module, we wrap our code into a simple-usage interface for using the storm identification and tracking functionalities.

    This class provides a high-level interface for identifying storms in radar images and tracking them over time. It abstracts away the details of storm identification and matching, allowing users to easily process radar data and obtain storm tracks.

    Main Attributes:
        dbz_threshold (int): The reflectivity threshold for storm identification.
        max_velocity (float): The maximum velocity for storm tracking.
        filter_area (float): The minimum area for filtering identified storms.

    Optional Attributes: see in class OursPrecipitationModel
    """
    model: OursPrecipitationModel

    def __init__(self, dbz_threshold: float = 35, max_velocity: float = 100, filter_area: float = 20, **args):
        self.dbz_threshold = dbz_threshold
        self.max_velocity = max_velocity
        self.filter_area = filter_area

        identifier = args.pop("identifier", None)
        if identifier is None:
            identifier = MorphContourIdentifier()

        model_kwargs = {}
        allowed_kwargs = {
            "weights",
            "radii",
            "num_sectors",
            "density",
            "velocity_estimate_weights",
            "particle_matching_method",
        }

        for key in allowed_kwargs:
            if key in args:
                model_kwargs[key] = args.pop(key)

        if args:
            raise TypeError(f"Unexpected keyword arguments for OursPrecipitationModel: {', '.join(sorted(args.keys()))}")

        self.model = OursPrecipitationModel(
            identifier=identifier,
            max_velocity=max_velocity,
            **model_kwargs,
        )

    def process_radar_image(self, dbz_img: np.ndarray, time_frame: datetime) -> StormsMap:
        """
        Processs an incoming radar image and save in the history of the model.

        Args:
            dbz_img (np.ndarray): The radar reflectivity image.
            time_frame (datetime): The timestamp of the radar image.

        Returns:
            StormsMap: The identified storms map.

        """
        if len(self.model.storms_maps) > 0:
            last_map_time_frame = self.model.storms_maps[-1].time_frame
            if time_frame <= last_map_time_frame:
                raise ValueError(f"Time frame of the new radar image ({time_frame}) must be greater than the last processed time frame ({last_map_time_frame}).")
        
        map_id = f"map_{len(self.model.storms_maps)}"

        storms_map = self.model.identify_storms(
            dbz_img=dbz_img,
            time_frame=time_frame,
            map_id=map_id,
            threshold=self.dbz_threshold,
            filter_area=self.filter_area,
        )
        self.model.processing_map(storms_map)

        return storms_map
    
    def print_maps(self):
        """
            Print the current history of storms maps in the model.
        """
        print("-"*20)
        print(f"Printing {len(self.model.storms_maps)} storms maps in the model history:")
        for idx, storms_map in enumerate(self.model.storms_maps):
            print(f"\tMap [{idx}]: Time Frame = {storms_map.time_frame}, Number of Storms = {len(storms_map.storms)}")
        print("-"*20)

    def forecast(self, lead_time: float) -> np.ndarray:
        """
            Forecast the future state of the storms after a given lead time.

            Args:
                lead_time (float): The lead time in hours for forecasting.

            Returns:
                np.ndarray: The forecasted storm map after the given lead time. This is binary map where 1 indicates the presence of a storm and 0 indicates no storm.
        """
        if self.model.tracker is None:
            raise ValueError("No storms have been processed yet. Please process at least one radar image before forecasting.")
        
        predicted_storms_map = self.model.forecast(lead_time)
        assert predicted_storms_map is not StormsMap, "The forecasted result must be a StormsMap instance."

        active_map = np.zeros_like(self.model.storms_maps[0].dbz_map, dtype=np.uint8)

        for storm in predicted_storms_map.storms:
            polygon_coords = np.asarray(storm.contour.exterior.coords, dtype=np.int32)
            polygon_coords = polygon_coords.reshape((-1, 1, 2))
            cv2.fillPoly(active_map, [polygon_coords], 1)

        return active_map
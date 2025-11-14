from dataclasses import dataclass
import numpy as np
from datetime import datetime
from shapely.affinity import translate

from src.tracking import BaseTrackingHistory
from src.cores.base import StormsMap

from .storm import CentroidStorm

@dataclass
class Correspondence:
    prev_indices: list[int]
    curr_indices: list[int]

class TrackingHistory(BaseTrackingHistory):
    def __init__(self, storms_map: StormsMap):
        self.tracks = [self._initialize_track(storm, storms_map.time_frame) for storm in storms_map.storms]
        self.storm_dict = {storm.id: idx for idx, storm in enumerate(storms_map.storms)}
        self.active_list = list(range(len(self.storm_dict)))
    
    def _initialize_track(self, storm: CentroidStorm, time_frame: datetime):
        return {"storm_lst": [storm], "frame": [time_frame], "movement": []}
    
    def _get_track(self, storm_id: str) -> tuple[dict, float]:
        """
        Get the track of storm with storm_id, if there is no track found, raise `KeyError`.

        Args:
            storm_id (str): id of the storm
        
        Returns:
            track, is_active (tuple(dict, float))
        """
        if storm_id not in self.storm_dict:
            raise KeyError(f"Storm {storm_id} not found in the current track.")
        track_id = self.storm_dict[storm_id]
        return self.tracks[track_id], track_id in self.active_list

    def _interpolate_velocity(self, velocity_lst: list[np.ndarray], alpha_decay: float = 0.5):
        if len(velocity_lst) == 1:
            return velocity_lst[0]
        
        weights = np.array([alpha_decay**i for i in range(len(velocity_lst))])
        total_w = np.sum(weights)
        return np.sum([displ * w / total_w for displ, w in zip(velocity_lst[::-1], weights)], axis=0)

    def forecast(self, storm_id: str, dt: float, default_motion: np.ndarray = np.array([0,0])) -> CentroidStorm:
        """
        Make a forecast for the next position of the track with track_id using the history.
        
        Args:
            storm_id (int): id of the storm.
            dt (float): the interval between the current and next frame.
            default_motion (np.ndarray, default): default motion used in case there is no recorded history.
        
        Returns:
            storm (CentroidStorm): the estimated storm in the next frame.
        """
        track, is_active = self._get_track(storm_id)
        if not is_active:
            print(f"⚠️ Storm has been expired")
            
        curr_storm = track["storm_lst"][-1]

        velocity_lst = track["movement"]
        if len(velocity_lst) == 0:      # if no recorded velocity => use the default motion.
            velocity_lst = [default_motion]
        
        dx, dy = self._interpolate_velocity(velocity_lst) * dt
        new_pol = translate(curr_storm.contour, xoff=dx, yoff=dy)
        new_centroid = curr_storm.centroid + np.array(dx, dy)

        return CentroidStorm(new_pol, centroid=new_centroid)
    
    def get_history_velocities(self, storm_lst: list[CentroidStorm]) -> list[np.ndarray]:
        """
        Get the historical velocities for the given storm list.

        Args:
            storm_lst (list[CentroidStorm]): List of storms to get historical velocities.
        Returns:
            list[np.ndarray]: List of historical velocities.
        """
        history_velocities = []
        for storm in storm_lst:
            track, _ = self._get_track(storm.id)
            if len(track["movement"]) == 0:
                history_velocities.append(None)
            else:
                history_velocities.append(np.array(self._interpolate_velocity(track["movement"])))
        return history_velocities

    def update(self, correspondence_lst: list[Correspondence], prev_storms_map: StormsMap, curr_storms_map: StormsMap):
        active_lst = []
        curr_time = curr_storms_map.time_frame
        prev_time = prev_storms_map.time_frame

        dt = (curr_time - prev_time).seconds / 3600

        def get_movement(prev_idx, curr_idx):
            """
            Get the centroid movement of 2 storms, scaled to pixel/hr.
            """
            return (curr_storms_map.storms[curr_idx].centroid - prev_storms_map.storms[prev_idx].centroid) / dt
        
        for corr in correspondence_lst:
            # case 1: new storm
            if len(corr.prev_indices) == 0:
                curr_storm = curr_storms_map.storms[corr.curr_indices[0]]
                # create a new track
                self.tracks.append(self._initialize_track(curr_storm, curr_time))
                new_uid = len(self.tracks) - 1
                self.storm_dict[curr_storm.id] = new_uid
                active_lst.append(new_uid)
            
            # case 2: only one curr storm
            elif len(corr.curr_indices) == 1:
                curr_storm = curr_storms_map.storms[corr.curr_indices[0]]
                prev_idx = corr.prev_indices[0]
                prev_storm = prev_storms_map.storms[prev_idx]
                track_id = self.storm_dict[prev_storm.id]
                track = self.tracks[track_id]
                track["storm_lst"].append(curr_storm)
                track["frame"].append(curr_time)
                track["movement"].append(get_movement(prev_idx, corr.curr_indices[0]))
                self.storm_dict[curr_storm.id] = track_id
                active_lst.append(track_id)
            
            # case 3: more than one curr storm => split
            else:
                # assign the UID of the parent to the child with least disparity (1st curr storm)
                prev_idx = corr.prev_indices[0]
                prev_storm = prev_storms_map.storms[prev_idx]
                
                ## resolve last track carry
                first_curr_idx = corr.curr_indices[0]
                curr_storm = curr_storms_map.storms[first_curr_idx]

                track_id = self.storm_dict[prev_storm.id]
                track = self.tracks[track_id]
                track["storm_lst"].append(curr_storm)
                track["frame"].append(curr_time)
                track["movement"].append(get_movement(prev_idx, first_curr_idx))
                self.storm_dict[curr_storm.id] = track_id
                active_lst.append(track_id)

                ## resolve new tracks for the rest
                for curr_idx in corr.curr_indices[1:]:
                    curr_storm = curr_storms_map.storms[curr_idx]
                    # create a new track
                    self.tracks.append(self._initialize_track(curr_storm, curr_time))
                    new_uid = len(self.tracks) - 1
                    self.storm_dict[curr_storm.id] = new_uid
                    active_lst.append(new_uid)
        
        self.active_list = active_lst
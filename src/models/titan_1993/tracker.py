from datetime import datetime
import numpy as np
from shapely.affinity import translate
from copy import deepcopy

from src.tracking import BaseTrackingHistory
from src.cores.base import StormsMap

from .storm import CentroidStorm

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
            raise KeyError(f"Storm not found in the current track.")
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

    def _handle_merge(self, merge_lst: list[dict]) -> np.ndarray:
        """
        Combine the list of storms to generate the parent storm history.
        
        Args:
            merge_lst (list[dict]): contains information about the storm, including movement history, area.
        """
        weights = np.array([s["area"] for s in merge_lst])
        movements_lst = [np.array(s["movement"])[::-1] for s in merge_lst]  # revert for matching the time.

        combined_len = max([len(movements) for movements in movements_lst]) # length of parent = max length of its child
        parent_movement = np.zeros(shape=(combined_len, 2), dtype=np.float64)

        # for each time, if the
        for i in range(combined_len):
            total_value, total_weight = np.zeros(shape=(2,)), 0
            for weight, movements in zip(weights, movements_lst):
                if len(movements) <= i:
                    continue
                total_value += movements[i]
                total_weight += weight
            parent_movement[i] = total_value / total_weight
        
        return parent_movement[::-1]
    
    def _handle_split(self):
        pass

    def update(
            self, prev_mapping: dict, curr_mapping: dict, prev_storms_map: StormsMap, curr_storms_map: StormsMap,
        ):
        """
        Update the tracking history using the new mapping data.

        Args:
            prev_mapping (dict): key -> idx of prev storm; items -> list[idx of curr storm].
            curr_mapping (dict): key -> idx of curr storm; items -> list[idx of prev storm].
            time_frame (datetime): time of current mapping.
        """
        active_lst = []         # update the new active list
        curr_time = curr_storms_map.time_frame
        prev_time = prev_storms_map.time_frame

        dt = (curr_time - prev_time).seconds / 3600

        def get_movement(prev_idx, curr_idx):
            """
            Get the centroid movement of 2 storms, scaled to pixel/hr.
            """
            return (curr_storms_map.storms[curr_idx].centroid - prev_storms_map.storms[prev_idx].centroid) / dt

        for curr_idx, matched in curr_mapping.items():
            curr_storm = curr_storms_map.storms[curr_idx]

            ## Case 1: no previous matching => create the new track.
            if len(matched) == 0:
                # create new track
                self.tracks.append(self._initialize_track(curr_storm, curr_time))

                # update storm dict and active_lst
                new_tid = len(self.tracks) - 1
                self.storm_dict[curr_storm.id] = new_tid
                active_lst.append(new_tid)
            
            # Case 2: more than 1 parent storms => merged
            elif len(matched) > 1:
                merge_lst = []
                max_area = 0
                track_id = None

                for prev_idx in matched:
                    prev_storm = prev_storms_map.storms[prev_idx]   # get the previous storm

                    track = self._get_track(prev_storm.id)[0]          # get the corresponding track
                    area = prev_storm.contour.area
                    if area > max_area:
                        max_area = area
                        track_id = self.storm_dict[prev_storm.id]   # track with highest area will be extended, others are terminated
                    
                    merge_lst.append({
                        "area": prev_storm.contour.area,
                        "movement": track["movement"] + [get_movement(prev_idx, curr_idx)]
                    })
                
                current_track = self.tracks[track_id]
                
                movement_history = self._handle_merge(merge_lst)    # resolve the history
                current_track["storm_lst"].append(curr_storm)
                current_track["movement"] = list(movement_history)  # update the movement
                current_track["frame"].append(curr_time)

                # update storm dict & active lst
                active_lst.append(track_id)
                self.storm_dict[curr_storm.id] = track_id
            
            # Case 3: only 1 parent storm
            else:
                prev_idx = matched[0]
                prev_storm = prev_storms_map.storms[prev_idx]   # get the previous storm

                # case 3.1: parent has more than 2 childrens => split
                #   => movement = combined centroid - previous storm centroid
                if len(prev_mapping[prev_idx]) > 1:
                    # get the combined centroid using area as weight.
                    weight_centroids = [(curr_storms_map.storms[j].centroid, curr_storms_map.storms[j].contour.area) \
                                          for j in prev_mapping[prev_idx]]
                    total_w = np.sum([w for _, w in weight_centroids])
                    combined_centroid = np.sum([centroid * w / total_w for centroid, w in weight_centroids], axis=0)

                    # get the movement from parent -> combined centroid
                    movement = (combined_centroid - prev_storm.centroid) / dt

                    # copy the previous track into the new, then update parameters.
                    new_track = deepcopy(self._get_track(prev_storm.id)[0])
                    self.tracks.append(new_track)
                    new_track["storm_lst"].append(curr_storm)
                    new_track["movement"].append(movement)
                    new_track["frame"].append(curr_time)

                    new_tid = len(self.tracks) - 1
                    self.storm_dict[curr_storm.id] = new_tid
                    active_lst.append(new_tid)

                # case 3.2: one-to-one parent-child
                else:
                    # update the current track
                    movement = get_movement(prev_idx, curr_idx)
                    current_track = self._get_track(prev_storm.id)[0]
                    current_track["storm_lst"].append(curr_storm)
                    current_track["movement"].append(movement)
                    current_track["frame"].append(curr_time)

                    track_id = self.storm_dict[prev_storm.id]
                    self.storm_dict[curr_storm.id] = track_id
                    active_lst.append(track_id)

        self.active_list = sorted(active_lst)
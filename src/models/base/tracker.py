from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import warnings
import numpy as np

from src.cores.base import StormObject, StormsMap

@dataclass
class StormTrack:
    id: int
    storms: dict[datetime, StormObject]
    start_frame: datetime
    merged_to: Optional['StormTrack'] = None                        # ID of the storm this storm merged into
    splitted_from: Optional['StormTrack'] = None                    # ID of the storm this storm splitted from

class UpdateType(Enum):
    MATCHED = 0
    MERGED = 1
    SPLITTED = 2
    NEW = 3

class TrackingHistory:
    tracks: list[StormTrack] = field(default_factory=list)
    storms_dict: dict[int, int] = field(default_factory=dict) # Mapping from storm ID to track ID

    def __init__(self, storms_map: StormsMap):
        self.tracks = []
        self.storms_dict = dict()
        for storm in storms_map.storms:
            self.add_new_track(storm, storms_map.time_frame)
    
    def _interpolate_velocity(self, velocity_lst: list[np.ndarray], alpha_decay: float = 0.5):
        """
        Interpolate the velocity using weighted average with decay factor alpha_decay.

        Args:
            velocity_lst (list[np.ndarray]): list of velocity vectors.
            alpha_decay (float, default=0.5): the decay factor.

        Returns:
            interpolated_velocity (np.ndarray): the interpolated velocity.
        """
        if len(velocity_lst) == 1:
            return velocity_lst[0]
        
        weights = np.array([alpha_decay**i for i in range(len(velocity_lst))])
        total_w = np.sum(weights)
        return np.sum([displ * w / total_w for displ, w in zip(velocity_lst[::-1], weights)], axis=0)

    def _handle_merge(self):
        pass

    def add_new_track(self, new_storm: StormObject, time_frame: datetime):
        """
        Add a new StormTrack to the tracking history and update the storms dictionary.
        """
        id = len(self.tracks)
        new_track = StormTrack(
            id=id,
            storms={time_frame: new_storm},
            start_frame=time_frame
        )
        self.tracks.append(new_track)
        self.storms_dict[new_storm.id] = id

    def update_track(self, prev_storm: StormObject, curr_storm: StormObject, update_type: UpdateType, 
                    time_frame: datetime, velocity: Optional[np.ndarray] = None):
        """
        Update an existing StormTrack with a new storm observation.
        """
        track_id = self.storms_dict.get(prev_storm.id)

        if track_id is None:
            raise ValueError(f"Previous storm ID {prev_storm.id} not found in tracking history.")
        
        # Get the Storm Track to update
        track = self.tracks[track_id]

        if update_type == UpdateType.MATCHED:               # Case 1 - Matched: Simply append the new storm to the existing track
            track.storms[time_frame] = curr_storm
            self.storms_dict[curr_storm.id] = track_id
            curr_storm.track_history(prev_storm, velocity)
            
        elif update_type == UpdateType.MERGED:              # Case 2 - Merged: Append the new storm and mark the track as merged
            curr_track_id = self.storms_dict.get(curr_storm.id)
            track.merged_to = self.tracks[curr_track_id]
        else:                                               # Case 3 - Splitted: Create a new track for the splitted storm with a reference to the original track
            id = len(self.tracks)
            new_track = StormTrack(
                id=id,
                storms={time_frame: curr_storm},
                start_frame=time_frame,
                splitted_from=track
            )
            self.tracks.append(new_track)
            self.storms_dict[curr_storm.id] = id
            curr_storm.track_history(prev_storm, velocity)

    def get_movement(self, storm_id: str, time_frame: datetime) -> list[np.ndarray]:
        """
        Get the list of recorded movements for the storm with storm_id.

        Args:
            storm_id (int): id of the storm.

        Returns:
            movement_lst (list[np.ndarray]): list of recorded movements.
        """
        track_id = self.storms_dict[storm_id]
        track = self.tracks[track_id]
            
        curr_storm: StormObject = track.storms[time_frame]

        history_movements = curr_storm.history_movements
        if len(history_movements) == 0:
            warnings.warn(f"No recorded movement for storm {storm_id} at time {time_frame}. Returning [None, None].")
            return [None, None]
        
        return self._interpolate_velocity(curr_storm.history_movements)
    
    def forecast(self, storm_id: str, dt: float, default_motion: np.ndarray = np.array([0,0])) -> StormObject:
        """
        Make a forecast for the next position of the track with track_id using the history.
        
        Args:
            storm_id (int): id of the storm.
            dt (float): the interval between the current and next frame.
            default_motion (np.ndarray, default): default motion used in case there is no recorded history.
        
        Returns:
            storm (CentroidStorm): the estimated storm in the next frame.
        """
        track_id = self.storms_dict[storm_id]
        track = self.tracks[track_id]
            
        curr_storm: StormObject = track.storms[max(track.storms.keys())]

        velocity_lst = curr_storm.history_movements
        if len(velocity_lst) == 0:      # if no recorded velocity => use the default motion.
            velocity_lst = [default_motion]
        
        displacement = self._interpolate_velocity(velocity_lst) * dt
        new_storm = curr_storm.copy()
        new_storm.make_move(displacement)

        return new_storm
    
    def print_tracks(self):
        """
        Print the tracking history for debugging.
        """
        for track in self.tracks:
            print(f"Track {track.id:2d}. " + " -> ".join(storm.id for storm in track.storms.values()))
            print(f"Lifespan: {len(track.storms)}")
            for storm in track.storms.values():
                history_movements = [f"({mv[0]:.2f}, {mv[1]:.2f})" for mv in storm.history_movements]
                print(f"\t{storm.id}: " + ", ".join(history_movements))
            print(f"-" * 50)
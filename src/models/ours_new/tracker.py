from copy import deepcopy
import numpy as np
from datetime import datetime

from src.cores.base import StormsMap
from src.tracking import BaseTrackingHistory
from .storm import ShapeVectorStorm
from dataclasses import dataclass, field
import warnings


@dataclass
class StormRecord:
    """
    Record: is a part of `StormTrack`, contains information about a storm at a specific time frame.
    """
    track_id: int
    storms: dict[int, ShapeVectorStorm]
    time_frame: datetime
    is_virtual: bool = field(default=False)      # if virtual: the current storm of the corresponding track is splitted and this storm is created to keep track of the splitted part.
    parent_track_id: int = field(default=None)   # if virtual: the id of the parent track from which this storm is splitted.

@dataclass
class StormTrack:
    """
    Track: contains the history of a storm over time frames.
    """
    id: int
    records: list[StormRecord] = field(default_factory=list)
    movements: list[np.ndarray] = field(default_factory=list)   # list of movement vectors over time
    active: bool = field(default=True)                          # whether the track is still active or not
    merged: bool = field(default=False)                         # in case the track is inactive, check whether the track has been merged into another track
    merged_to: int = field(default=None)                        # if merged, the id of the track to which this track is merged

    def add_record(self, record: StormRecord, motion_vector: np.ndarray = None):
        self.records.append(record)
        if motion_vector is not None:
            self.movements.append(motion_vector)
    
    def get_latest_record(self) -> StormRecord:
        return self.records[-1] if self.records else None
    
    def get_copy(self, new_id: int, last_time: datetime) -> "StormTrack":
        """
        Return a copy of the current track with a new id.
        """
        new_track = StormTrack(id=new_id)
        new_track.records = deepcopy(self.records)
        for record in new_track.records:
            record.track_id = new_id
            record.is_virtual = True
            record.parent_track_id = self.id

        for movement in self.movements:
            new_track.movements.append(deepcopy(movement))

        if len(new_track.records) > 0 and new_track.records[-1].time_frame >= last_time:
            # drop the last record if it exceeds the last time
            new_track.records.pop()
            if len(new_track.movements) > 0:
                new_track.movements.pop()

        return new_track

class TrackingHistory(BaseTrackingHistory):
    tracks: list[StormTrack] = []
    storm_dict: dict[str, int] = {}      # mapping from storm id to track id
    active_list: list[int] = []          # list of active track ids

    def __init__(self, storms_map: StormsMap):
        self.tracks = []
        for storm in storms_map.storms:
            self.tracks.append(self._initialize_track(storm, storms_map.time_frame))
        
        self.storm_dict = {storm.id: idx for idx, storm in enumerate(storms_map.storms)}
        self.active_list = list(range(len(self.tracks)))
    
    def _initialize_track(self, storm: ShapeVectorStorm, time_frame: datetime):
        new_id = len(self.tracks)
        return StormTrack(
            id=new_id,
            records=[StormRecord(
                track_id=new_id,
                storm=storm,
                time_frame=time_frame
            )]
        )

    def _get_track(self, storm_id: str) -> StormTrack:
        """
        Get the track of storm with storm_id, if there is no track found, raise `KeyError`.

        Args:
            storm_id (str): id of the storm
        
        Returns:
            track (dict): the track information of the storm.
        """
        if storm_id not in self.storm_dict:
            raise KeyError(f"Storm {storm_id} not found in the current track.")
        track_id = self.storm_dict[storm_id]
        return self.tracks[track_id]

    def _interpolate_velocity(self, velocity_lst: list[np.ndarray], alpha_decay: float = 0.5):
        if len(velocity_lst) == 1:
            return velocity_lst[0]
        
        weights = np.array([alpha_decay**i for i in range(len(velocity_lst))])
        total_w = np.sum(weights)
        return np.sum([displ * w / total_w for displ, w in zip(velocity_lst[::-1], weights)], axis=0)
    
    def forecast(self, storm_id: str, dt: float, default_motion: np.ndarray = np.array([0,0])) -> ShapeVectorStorm:
        """
        Make a forecast for the next position of the track with track_id using the history.
        
        Args:
            storm_id (int): id of the storm.
            dt (float): the interval between the current and next frame.
            default_motion (np.ndarray, default): default motion used in case there is no recorded history.
        
        Returns:
            storm (CentroidStorm): the estimated storm in the next frame.
        """
        track = self._get_track(storm_id)
        if not track.active:
            warnings.warn(f"Storm {storm_id} is expired. Forecast might be inaccurate.")
            
        curr_storm: ShapeVectorStorm = track.records[-1].storm

        velocity_lst = track.movements
        if len(velocity_lst) == 0:      # if no recorded velocity => use the default motion.
            velocity_lst = [default_motion]
        
        displacement = self._interpolate_velocity(velocity_lst) * dt
        new_storm = curr_storm.copy()
        new_storm.make_move(displacement)

        return new_storm

    def update(
            self, mapping_curr: dict[int, list], inherited_dict: dict[int, dict], prev_storms_map: StormsMap, curr_storms_map: StormsMap
        ):
        """
        Update the tracking history using the new mapping data.

        Args:
            curr_matching_list (list): list of matched storms information where list's indices are storm indices in the current storms map.
            time_frame (datetime): time of current mapping.
        """
        active_lst = []         # update the new active list
        merge_lst = []          # list of merged storms information to be handled later.
        curr_time = curr_storms_map.time_frame
        # prev_time = prev_storms_map.time_frame

        # dt = (curr_time - prev_time).seconds / 3600

        def update_track(prev_storm_order: int, curr_storm_order: int, motion_vector: np.ndarray) -> int:
            """
            Update the track of the current assignment and return the track id.
            """
            prev_storm_idx = prev_storms_map.storms[prev_storm_order].id
            curr_storm = curr_storms_map.storms[curr_storm_order]
            curr_storm_idx = curr_storm.id

            # check if the parent track is virtual (due to splitting) or not
            virtual = inherited_dict[curr_storm_order].get("virtual", None)
            if virtual is None:
                warnings.warn(f"Cannot find inherited information for storm {curr_storm_order}. Assuming parent track is not virtual.")

            ## case *.1: parent track is not virtual => update into existing track
            if virtual is False:
                # print(f"update existing track for storm {curr_storm.id} from parent storm {prev_storms_map.storms[prev_storm_order].id}")
                parent_track = self._get_track(prev_storm_idx)

                parent_track.add_record(
                    StormRecord(
                        track_id=parent_track.id,
                        storm=curr_storm,
                        time_frame=curr_time,
                    ),
                    motion_vector=motion_vector
                )

                self.storm_dict[curr_storm_idx] = parent_track.id
                return parent_track.id
            
            ## case *.2: parent track is virtual => create a new track from the parent track
            else:
                parent_track = self._get_track(prev_storm_idx)
                new_id = len(self.tracks)
                new_track = parent_track.get_copy(new_id=new_id, last_time=curr_time)
                new_track.add_record(StormRecord(
                        track_id=new_id,
                        storm=curr_storm,
                        time_frame=curr_time,
                    ),
                    motion_vector=motion_vector
                )

                self.tracks.append(new_track)
                self.storm_dict[curr_storm_idx] = new_id
                
                return new_id

        for curr_storm_order, matched_list in mapping_curr.items():
            curr_storm = curr_storms_map.storms[curr_storm_order]
            matched_list = sorted(matched_list, key=lambda x: x[2], reverse=True)

            # case 1: no matched storms => create new track
            if len(matched_list) == 0:
                new_track = self._initialize_track(
                    curr_storm, curr_time
                )
                self.tracks.append(new_track)
                new_id = len(self.tracks) - 1
                self.storm_dict[curr_storm.id] = new_id
                active_lst.append(new_id)
            
            # case 2: only one matched storm => continue the track
            elif len(matched_list) == 1:
                prev_storm_order, _, _, motion_vector = matched_list[0]

                track_id = update_track(prev_storm_order, curr_storm_order, motion_vector)
                active_lst.append(track_id)
            
            # case 3: multiple matched storms => mark as merged for later handling
            else:
                # update the parent storm into the main track
                prev_storm_order, _, _, motion_vector = matched_list[0]   # get the one with highest number of matched particles
                track_id = update_track(prev_storm_order, curr_storm_order, motion_vector)
                active_lst.append(track_id)

                # for others, mark them as 'merged'
                for prev_storm_order, _, _, motion_vector in matched_list[1:]:   # sort by number of matched particles
                    prev_storm_idx = prev_storms_map.storms[prev_storm_order].id
                    merged_track_id = self.storm_dict[prev_storm_idx]
                    track = self.tracks[merged_track_id]
                    track.merged_to = track_id

                    merge_lst.append(merged_track_id)

        # update active status
        for id in self.active_list:
            if id not in active_lst:
                self.tracks[id].active = False

            if id in merge_lst:
                self.tracks[id].merged = True
        
        self.active_list = sorted(active_lst)

    def print_tracks(self):
        for id, track in enumerate(self.tracks):
            if track.active:
                status = "ACTIVE"
            elif track.merged:
                status = f"MERGED to track {track.merged_to}"
            else:
                status = "EXPIRED"

            print(f"Track {id:2d}. " + " -> ".join(record.storm.id for record in track.records))
            print(f"Id: {track.id}")
            print(f"Status: {status}")
            print(f"Lifespan: {len(track.records)}")
            print(f"Last track: {track.records[-1].time_frame.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"-" * 50)
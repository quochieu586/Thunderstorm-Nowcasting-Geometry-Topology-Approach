from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

from src.cores.base import StormObject

@dataclass
class StormTrack:
    id: int
    storms: dict[datetime, StormObject]
    start_frame: datetime
    merged_to: Optional['StormTrack'] = None                        # ID of the storm this storm merged into
    splitted_from: Optional['StormTrack'] = None                    # ID of the storm this storm splitted from

class UpdateType(Enum):
    MATCHED = "matched"
    MERGED = "merged"
    SPLITTED = "splitted"

class TrackingHistory:
    tracks: list[StormTrack] = field(default_factory=list)
    storms_dict: dict[int, int] = field(default_factory=dict) # Mapping from storm ID to track ID

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

    def update_track(self, prev_storm: StormObject, curr_storm: StormObject, update_type: UpdateType, time_frame: datetime):
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

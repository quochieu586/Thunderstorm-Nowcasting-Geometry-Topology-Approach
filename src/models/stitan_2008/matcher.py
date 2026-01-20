import numpy as np

from src.cores.movement_estimate import TREC
from src.cores.base import StormsMap, StormObject

from ..base.tracker import MatchedStormPair, UpdateType
from .storm import ParticleStorm

class STitanMatcher:
    trec: TREC

    def __init__(self, trec: TREC):
        self.trec = trec
    
    def match_storms(
            self, storms_map_1: StormsMap, storms_map_2: StormsMap, 
            matching_overlap_threshold: float = 0.5     # for TREC overlapping matching
        ) -> list[MatchedStormPair]:
        """
        Match storms between 2 time frame.

        Args:
            storms_map_1 (StormsMap): storm map in the 1st frame.
            storms_map_2 (StormsMap): storm map in the 2nd frame.
        
        Returns:
            assignments (np.ndarray): Array of (prev_idx, curr_idx) pairs representing matched storms.
        """
        grid_y, grid_x, vy, vx = self.trec.estimate_movement(storms_map_1, storms_map_2)

        # Count the number of forecasted particles in each storm
        NF_matrix = np.zeros((len(storms_map_1.storms), len(storms_map_2.storms)), dtype=np.float64)
        for prev_idx, prev_storm in enumerate(storms_map_1.storms):
            prev_storm: ParticleStorm = prev_storm  # type: ignore
            
            forecasted_particles = prev_storm.forecast_particles(grid_y, grid_x, vy, vx)
            
            for curr_idx, curr_storm in enumerate(storms_map_2.storms):
                curr_storm: ParticleStorm = curr_storm  # type: ignore
                count_in_storm = curr_storm.count_particles_in_storm(
                    forecasted_particles, storms_map_1.dbz_map.shape
                )
                NF_matrix[prev_idx, curr_idx] = count_in_storm / min(prev_storm.get_num_particles(), \
                                                                     curr_storm.get_num_particles())
            
        temp_assignments = np.where(NF_matrix >= matching_overlap_threshold)

        # Resolve split & merge for cases where one-to-many or many-to-one matching occurs
        assignments: list[MatchedStormPair] = []
        prev_correspondence = {prev_idx: [] for prev_idx in range(len(storms_map_1.storms))}
        curr_correspondence = {curr_idx: [] for curr_idx in range(len(storms_map_2.storms))}

        ## Keep tracks of both matched storm and its NF scor
        for prev_idx, curr_idx in zip(temp_assignments[0], temp_assignments[1]):
            prev_correspondence[prev_idx].append(curr_idx)
            curr_correspondence[curr_idx].append(prev_idx)

        for prev_idx in prev_correspondence.keys():
            # max NF first
            prev_correspondence[prev_idx].sort(
                key=lambda curr_idx: NF_matrix[prev_idx, curr_idx], reverse=True
            )
        
        for curr_idx in curr_correspondence.keys():
            # max NF first
            curr_correspondence[curr_idx].sort(
                key=lambda prev_idx: NF_matrix[prev_idx, curr_idx], reverse=True
            )
        
        ## Finalize assignments
        assigned_prev = set()
        assigned_curr = set()

        for curr_idx in curr_correspondence.keys():
            curr_storm = storms_map_2.storms[curr_idx]

            # case 1: new storm
            if len(curr_correspondence[curr_idx]) == 0:
                assignments.append(
                    MatchedStormPair(
                        prev_storm_order=-1,
                        curr_storm_order=curr_idx,
                        estimated_movement=None,
                        update_type=UpdateType.NEW
                    )
                )
                continue
            
            # get the movement by combining movements of previous storms
            movement_vectors = []
            for prev_idx in curr_correspondence[curr_idx]:
                prev_storm = storms_map_1.storms[prev_idx]
                movement_vectors.append(
                    self.trec.get_storm_centroid_movement(
                        prev_storm, grid_y, grid_x, vy, vx
                    )
                )
            movement = np.mean(movement_vectors, axis=0) if len(movement_vectors) > 0 else None
            if movement is None:
                raise ValueError("Movement cannot be None here.")

            # case 2: already assigned => mark current as split
            if prev_idx in assigned_prev:   
                assignments.append(
                    MatchedStormPair(
                        prev_storm_order=prev_idx,
                        curr_storm_order=curr_idx,
                        estimated_movement=movement,
                        update_type=UpdateType.SPLITTED
                    )
                )
                assigned_curr.add(curr_idx)

            # case 3: not assigned yet => assign normally
            assignments.append(
                MatchedStormPair(
                    prev_storm_order=prev_idx,
                    curr_storm_order=curr_idx,
                    estimated_movement=movement,
                    update_type=UpdateType.MATCHED
                )
            )
            assigned_prev.add(prev_idx)
            assigned_curr.add(curr_idx)
        
        # check for merge case
        for prev_idx in prev_correspondence.keys():
            if prev_idx in assigned_prev or len(prev_correspondence[prev_idx]) < 1:
                continue
            
            curr_idx = prev_correspondence[prev_idx][0]
            if curr_idx in assigned_curr:
                assignments.append(
                    MatchedStormPair(
                        prev_storm_order=prev_idx,
                        curr_storm_order=curr_idx,
                        estimated_movement=np.array([0.0, 0.0]),
                        update_type=UpdateType.MERGED
                    )
                )
            
            assigned_prev.add(prev_idx)
        
        assignments.sort(key=lambda info: info.update_type.value)
        return assignments
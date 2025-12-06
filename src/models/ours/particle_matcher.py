from dataclasses import dataclass, field
import numpy as np
from enum import Enum

# from src.models.base.tracker import UpdateType
from src.cores.polar_description_vector import ShapeVector
from src.tracking import BaseMatcher
from .storm import ShapeVectorStorm

@dataclass
class Particle:
    feature: ShapeVector
    storm_id: str           # index of storm

@dataclass
class MatchedStormInfo:
    prev_storm_id: str
    curr_storm_id: str
    prev_score: float = field(default=None)
    curr_score: float = field(default=None)
    displacement_list: list = field(default_factory=list)

    def count_matches(self) -> int:
        """
        Count the number of matched particles.
        """
        return len(self.displacement_list)

    def append_displacement(self, displacement: np.ndarray):
        self.displacement_list.append(displacement)     # in (dy, dx) order
    
    def derive_displacement(self) -> np.ndarray:
        """
        Derive the average displacement from matched particles.
        """
        if self.count_matches() == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        return np.mean(np.array(self.displacement_list), axis=0)

class ParticleMatcher(BaseMatcher):
    def _construct_disparity_matrix(
            self, particle_lst1: list[Particle], particle_lst2: list[Particle], weights: list[float]
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct a disparity matrix for the 2 list of shape vector objects.

        Parameters:
            particle_lst1 (list[Particle]): The first list of shape vectors.
            particle_lst2 (list[Particle]): The second list of shape vectors.
            weights (list[float]): the list of non-negative and sum-to-1 weights.

        Returns:
            cost_matrix, distance_matrix (tuple[np.ndarray, np.ndarray]): The constructed disparity matrix and distance matrix for later verification.
        """
        assert all([all([w >= 0 for w in weights]), np.sum(weights) == 1]), "weights must not be negative and must sum to 1."
        coord_1 = np.array([np.array(p.feature.coord) for p in particle_lst1])
        coord_2 = np.array([np.array(p.feature.coord) for p in particle_lst2])
        distance_matrix = np.linalg.norm(coord_1[:, None, :] - coord_2[None, :, :], axis=2)

        shape_vector_1 = np.array([p.feature.vector for p in particle_lst1])
        shape_vector_2 = np.array([p.feature.vector for p in particle_lst2])
        shape_diff_matrix = np.sqrt(np.abs(shape_vector_1[:, None, :] - shape_vector_2[None, :, :]).sum(axis=2))

        return weights[0] * distance_matrix + weights[1] * shape_diff_matrix, distance_matrix

    def match_storms(
            self, prev_storms_list: list[ShapeVectorStorm], curr_storms_list: list[ShapeVectorStorm], weights: list[float],
            maximum_displacement: float, matching_threshold: float
        ) -> list[MatchedStormInfo]:
        """
        Match storms between 2 time frame.
        Steps:
        1. Construct particles from shape vectors of storms.
        2. Match particles using Hungarian algorithm based on disparity matrix. 
        3. Aggregate matched particles to form matched storms.
        """
        prev_order_mapping = {storm.id: idx for idx, storm in enumerate(prev_storms_list)}
        curr_order_mapping = {storm.id: idx for idx, storm in enumerate(curr_storms_list)}

        particles_prev: list[Particle] = [Particle(feature=v, storm_id=storm.id) for storm in prev_storms_list\
                          for v in storm.shape_vectors]
        particles_curr: list[Particle] = [Particle(feature=v, storm_id=storm.id) for storm in curr_storms_list\
                          for v in storm.shape_vectors]

        # particles matching
        cost_matrix, displacement_matrix = self._construct_disparity_matrix(particles_prev, particles_curr, weights=weights)
        invalid_mask = displacement_matrix > maximum_displacement

        cost_matrix = cost_matrix + invalid_mask.astype(np.float64) * 3000      # add penalty to those violated
        row_ind, col_ind = self._hungarian_matching(cost_matrix)

        assignment_mask = np.zeros_like(invalid_mask, dtype=bool)
        assignment_mask[row_ind, col_ind] = True

        particle_assignments = np.argwhere(assignment_mask & np.logical_not(invalid_mask))

        # resolve particle assignment to storm assignment
        matched_storms_dict = dict[tuple[str, str], MatchedStormInfo]()
        matched_info_list: list[MatchedStormInfo] = []

        # collect matched particles information for each pair of storms
        for idx1, idx2 in particle_assignments:
            prev_storm_id = particles_prev[idx1].storm_id
            curr_storm_id = particles_curr[idx2].storm_id

            if (prev_storm_id, curr_storm_id) not in matched_storms_dict:
                matched_storms_dict[(prev_storm_id, curr_storm_id)] = MatchedStormInfo(
                    prev_storm_id=prev_storm_id,
                    curr_storm_id=curr_storm_id
                )
            
            displacement = np.array(particles_curr[idx2].feature.coord) - np.array(particles_prev[idx1].feature.coord)
            matched_storms_dict[(prev_storm_id, curr_storm_id)].append_displacement(displacement)

        for (prev_storm_id, curr_storm_id), matched_info in matched_storms_dict.items():
            num_particles_prev = prev_storms_list[prev_order_mapping[prev_storm_id]].get_num_particles()
            num_particles_curr = curr_storms_list[curr_order_mapping[curr_storm_id]].get_num_particles()
            min_particles = min(num_particles_prev, num_particles_curr)

            score = matched_info.count_matches() / min_particles

            # only keep those matched storm pairs with score >= matching_threshold
            if score >= matching_threshold:
                matched_info.prev_score = matched_info.count_matches() / num_particles_prev
                matched_info.curr_score = matched_info.count_matches() / num_particles_curr
                matched_info_list.append(matched_info)

        return matched_info_list
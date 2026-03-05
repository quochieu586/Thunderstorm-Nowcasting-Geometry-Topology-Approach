import numpy as np

from src.cores.movement_estimate import TREC
from src.cores.base import StormsMap, StormObject
from src.tracking import reduced_soft_hungarian

from ..base.tracker import MatchedStormPair, UpdateType
from .storm import ParticleStorm
from .particle_matcher import ParticleMatcher, Particle
from .subset import SubsetResolver, CellSubset

class ISCITMatcher:
    trec: TREC

    def __init__(
            self, max_velocity: float, trec: TREC, weights: tuple[float, float], matching_threshold: float = 0.5
        ):
        self.max_velocity = max_velocity
        self.trec = trec
        self.weights = weights
        self.particles_matcher = ParticleMatcher()
        self.subset_resolver = SubsetResolver(matching_threshold)
    
    def match_storms(
            self, storms_map_1: StormsMap, storms_map_2: StormsMap, 
            first_guess: bool = False     # for TREC overlapping matching
        ) -> list[MatchedStormPair]:
        """
        Match storms between 2 time frame.

        Args:
            storms_map_1 (StormsMap): storm map in the 1st frame.
            storms_map_2 (StormsMap): storm map in the 2nd frame.
        
        Returns:
            assignments (list[MatchedStormPair]): Array of (prev_idx, curr_idx) pairs representing matched storms.
        """
        # get particles lists for both time frames
        particles_prev: list[Particle] = [Particle(position=particle, storm_order=idx) for idx, storm in enumerate(storms_map_1.storms) for particle in storm.particles]
        particles_curr: list[Particle] = [Particle(position=particle, storm_order=idx) for idx, storm in enumerate(storms_map_2.storms) for particle in storm.particles]

        # compute maximum displacement
        dt = (storms_map_2.time_frame - storms_map_1.time_frame).total_seconds() / 3600.0
        maximum_displacement = self.max_velocity * dt

        # estimate movement vectors for storms in the previous frame
        ## 1. Case first_guess = True: use TREC to estimate movement field
        estimated_particles_vectors = []
        estimated_storms_vectors = []

        if first_guess:
            grid_y, grid_x, vy, vx = self.trec.estimate_movement(storms_map_1, storms_map_2)
            for prev_storm in storms_map_1.storms:
                storm_vector = self.trec.average_storm_movement(
                    prev_storm, storms_map_1.dbz_map.shape[:2],
                    grid_y, grid_x, vy, vx
                )
                estimated_particles_vectors.extend([storm_vector] * len(prev_storm.particles))
                estimated_storms_vectors.append(storm_vector)
            
        ## 2. Case first_guess = False: use previous displacement to estimate movement vectors
        else:
            for prev_storm in storms_map_1.storms:
                movement = prev_storm.get_movement()
                if movement is None:
                    movement = np.array([0.0, 0.0])
                else:
                    movement = movement * dt
                estimated_particles_vectors.extend([movement] * len(prev_storm.particles))    # in (dy, dx) order, unit: pixels
                estimated_storms_vectors.append(movement)
        
        estimated_particles_vectors = np.array(estimated_particles_vectors)    # shape: (num_prev_storms, 2)

        # match particles
        particle_assignments = self.particles_matcher.match_particles(
            particles_prev, particles_curr, estimated_particles_vectors, maximum_displacement, self.weights
        )

        # count number of matched particles between each pair of storms
        matching_count = np.zeros((len(storms_map_1.storms), len(storms_map_2.storms)), dtype=np.int64)
        for prev_particle_idx, curr_particle_idx in particle_assignments:
            prev_storm_idx = particles_prev[prev_particle_idx].storm_order
            curr_storm_idx = particles_curr[curr_particle_idx].storm_order
            matching_count[prev_storm_idx, curr_storm_idx] += 1
        
        # compute probability matrix
        ## p_A: Probability of matching where the denominator is the number of particles in the previous storm
        ## p_B: Probability of matching where the denominator is the number of particles in the curr storm
        p_A = matching_count / np.array([len(storm.particles) for storm in storms_map_1.storms])[:, np.newaxis]
        p_B = matching_count / np.array([len(storm.particles) for storm in storms_map_2.storms])[np.newaxis, :]

        # find the soft-hungarian assignment
        assignments_A = reduced_soft_hungarian(1 - p_A)
        assignments_B = reduced_soft_hungarian(1 - p_B)
        ## temporary version
        temp_assignments = [(i, j) for i, j in list(set(assignments_A).union(set(assignments_B))) \
                            if (p_A[i, j] >= 0.1) and (p_B[i, j] >= 0.1)]

        ## justified version
        subsets = self.subset_resolver.create_subsets(temp_assignments)
        final_assignments: list[MatchedStormPair] = []
        for subset in subsets:
            estimated_particles_vectors = []
            for prev_idx in subset.prev_indices:
                prev_storm = storms_map_1.storms[prev_idx]
                estimated_particles_vectors.extend([estimated_storms_vectors[prev_idx]] * len(prev_storm.particles))
            
            estimated_particles_vectors = np.array(estimated_particles_vectors)    # shape: (num_prev_storms, 2)
            final_assignments.extend(
                self.subset_resolver.resolve_subset(
                    subset, storms_map_1, storms_map_2, estimated_particles_vectors,
                    self.max_velocity, self.weights
                )
            )

        # update for unmatched storms in the current frame
        curr_mapping = {curr_idx: False for curr_idx in range(len(storms_map_2.storms))}
        for match in final_assignments:
            curr_mapping[match.curr_storm_order] = True
        
        for curr_idx, mapped in curr_mapping.items():
            if not mapped:
                final_assignments.append(MatchedStormPair(
                    prev_storm_order=None,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.NEW,
                    estimated_movement=np.array([0.0, 0.0])
                ))

        return final_assignments
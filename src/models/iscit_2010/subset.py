import numpy as np
from dataclasses import dataclass

from src.tracking import BaseMatcher, reduced_soft_hungarian
from src.cores.base import StormsMap

from ..base.tracker import MatchedStormPair, UpdateType
from .particle_matcher import Particle, ParticleMatcher
from .storm import ParticleStorm

@dataclass
class CellSubset:
    """
    Keep track of assigned storms cluster.
    """
    prev_indices: set[int]
    curr_indices: set[int]

    def add_cell(self, prev_idx: int = None, curr_idx: int = None):
        """
        add a source/target storm index to the subcell.
        """
        if prev_idx is not None:
            self.prev_indices.add(prev_idx)
        if curr_idx is not None:
            self.curr_indices.add(curr_idx)

    def merge_subcell(self, other: "CellSubset"):
        """
        merge another subcell into this one.
        """
        self.prev_indices = self.prev_indices.union(other.prev_indices)
        self.curr_indices = self.curr_indices.union(other.curr_indices)

    def contains(self, storm_order: int, is_source: bool=True) -> bool:
        """
        check if the cell contains the given source/target storm index.
        """
        return (storm_order in self.prev_indices) if is_source else (storm_order in self.curr_indices)

class SubsetResolver:
    matching_threshold: float

    def __init__(self, matching_threshold: float):
        self.matching_threshold = matching_threshold

    def create_subsets(self, assignments: list[tuple[int, int]]) -> list[CellSubset]:
        """
        create subsets of storms based on the assignments between 2 time frames.
        """
        subcells: list[CellSubset] = []

        for prev_idx, curr_idx in assignments:
            prev_subcell = None
            curr_subcell = None

            # find if there is existing subcell containing prev_idx or curr_idx
            for subcell in subcells:
                if subcell.contains(prev_idx, is_source=True):
                    prev_subcell = subcell
                if subcell.contains(curr_idx, is_source=False):
                    curr_subcell = subcell
            
            # case 1: both are not belonged to any subcell => create new subcell
            if not prev_subcell and not curr_subcell:
                # create new subcell
                new_subcell = CellSubset(prev_indices={prev_idx}, curr_indices={curr_idx})
                subcells.append(new_subcell)
                continue
                
            # case 2: only one of them is belonged to a subcell => add the other index to that subcell
            if prev_subcell and not curr_subcell:
                prev_subcell.add_cell(curr_idx=curr_idx)
                continue
            if not prev_subcell and curr_subcell:
                curr_subcell.add_cell(prev_idx=prev_idx)
                continue
                
            # case 3: both are belonged to different subcells => merge the 2 subcells
            if prev_subcell != curr_subcell:
                prev_subcell.merge_subcell(curr_subcell)
                subcells.remove(curr_subcell)
        
        return subcells

    def resolve_subset(
            self, subset: CellSubset, 
            storms_map_lst_1: StormsMap, storms_map_lst_2: StormsMap, estimated_vectors: np.ndarray,
            max_velocity: float, weights: list[float] = [0.5, 0.5]
        ) -> list[MatchedStormPair]:
        """
        resolve the subset to get final assignments between 2 time frames.
        """
        # compute maximum displacement
        dt = (storms_map_lst_1.time_frame - storms_map_lst_2.time_frame).total_seconds() / 3600.0
        maximum_displacement = max_velocity * dt
        particles_matcher = ParticleMatcher()

        prev_storms_lst: list[tuple[int, ParticleStorm]] = [(idx, storm) for idx, storm in enumerate(storms_map_lst_1.storms) \
                                                            if idx in subset.prev_indices]
        curr_storms_lst: list[tuple[int, ParticleStorm]] = [(idx, storm) for idx, storm in enumerate(storms_map_lst_2.storms) \
                                                            if idx in subset.curr_indices]

        particles_prev: list[Particle] = [Particle(position=particle, storm_order=idx) for idx, storm in prev_storms_lst for particle in storm.particles]
        particles_curr: list[Particle] = [Particle(position=particle, storm_order=idx) for idx, storm in curr_storms_lst for particle in storm.particles]

        # match particles
        particle_assignments = particles_matcher.match_particles(
            particles_prev, particles_curr, estimated_vectors, maximum_displacement, weights
        )

        # count number of matched particles between each pair of storms
        matching_count = np.zeros(
            (len(storms_map_lst_1.storms), len(storms_map_lst_2.storms)), dtype=np.int64
        )

        for idx1, idx2 in particle_assignments:
            matching_count[particles_prev[idx1].storm_order, particles_curr[idx2].storm_order] += 1
        
        # compute probability matrix
        ## p_A: Probability of matching where the denominator is the number of particles in the previous storm
        ## p_B: Probability of matching where the denominator is the number of particles in the curr storm
        p_A = matching_count / np.array([len(storm.particles) for storm in storms_map_lst_1.storms])[:, np.newaxis]
        p_B = matching_count / np.array([len(storm.particles) for storm in storms_map_lst_2.storms])[np.newaxis, :]

        p = np.max([p_A, p_B], axis=0)
        assignments = np.argwhere(p >= self.matching_threshold)

        new_subsets = self.create_subsets(assignments)

        final_assignments: list[MatchedStormPair] = []
        assigned_prev = set()
        assigned_curr = set()

        for new_subset in new_subsets:
            prev_correspondence = {prev_idx: [] for prev_idx in new_subset.prev_indices}
            curr_correspondence = {curr_idx: [] for curr_idx in new_subset.curr_indices}

            for prev_idx, curr_idx in assignments:
                if prev_idx in new_subset.prev_indices and curr_idx in new_subset.curr_indices:
                    prev_correspondence[prev_idx].append(curr_idx)
                    curr_correspondence[curr_idx].append(prev_idx)

            ## 1. split
            for prev_idx in prev_correspondence.keys():
                # highest p_A => inherit parent
                prev_correspondence[prev_idx] = sorted(
                    prev_correspondence[prev_idx], key=lambda x: p_A[prev_idx, x], reverse=True
                )
            
            ## 2. merge
            for curr_idx in curr_correspondence.keys():
                # highest p_B => inherit parent
                curr_correspondence[curr_idx] = sorted(
                    curr_correspondence[curr_idx], key=lambda x: p_B[x, curr_idx], reverse=True
                )

            # Determine velocity = weighted average displacement
            def _get_weighted_centroid(storms: list[ParticleStorm]) -> np.ndarray:
                """
                Get the centroids of the list of storms.

                Args:
                    storms (list[ParticleStorm]): the list of storms.

                Returns:
                    centroids (np.ndarray): the array of centroids.
                """
                centroids = np.array([storm.centroid for storm in storms])
                areas = np.array([storm.contour.area for storm in storms])
                weights = areas / np.sum(areas)

                return np.average(centroids, weights=weights, axis=0)
        
            prev_mean_centroid = _get_weighted_centroid(
                [storms_map_lst_1.storms[idx] for idx in new_subset.prev_indices]
            )   # (y, x)
            curr_mean_centroid = _get_weighted_centroid(
                [storms_map_lst_2.storms[idx] for idx in new_subset.curr_indices]
            )   # (y, x)
            displacement = curr_mean_centroid - prev_mean_centroid   # (dy, dx)
            
            for curr_idx, prev_indices in curr_correspondence.items():
                if len(prev_indices) == 0:
                    final_assignments.append(
                        MatchedStormPair(
                            prev_storm_order=None,
                            curr_storm_order=curr_idx,
                            update_type=UpdateType.NEW,
                            estimated_movement=np.array([0.0, 0.0])
                        )
                    )
                    assigned_curr.add(curr_idx)
                    continue

                prev_idx = prev_indices[0]
                if prev_idx in assigned_prev:   # case split
                    final_assignments.append(
                        MatchedStormPair(
                            prev_storm_order=prev_idx,
                            curr_storm_order=curr_idx,
                            update_type=UpdateType.SPLITTED,
                            estimated_movement=displacement
                        )
                    )
                    assigned_curr.add(curr_idx)
                else:
                    final_assignments.append(
                        MatchedStormPair(
                            prev_storm_order=prev_idx,
                            curr_storm_order=curr_idx,
                            update_type=UpdateType.MATCHED,
                            estimated_movement=displacement
                        )
                    )
                    assigned_prev.add(prev_idx)
                    assigned_curr.add(curr_idx)
                
            for prev_idx, curr_indices in prev_correspondence.items():
                if prev_idx in assigned_prev or len(curr_indices) == 0:
                    continue

                curr_idx = curr_indices[0]
                if curr_idx in assigned_curr:   # case merge
                    final_assignments.append(
                        MatchedStormPair(
                            prev_storm_order=prev_idx,
                            curr_storm_order=curr_idx,
                            update_type=UpdateType.MERGED,
                            estimated_movement=displacement
                        )
                    )
                    assigned_prev.add(prev_idx)
                else:
                    final_assignments.append(
                        MatchedStormPair(
                            prev_storm_order=prev_idx,
                            curr_storm_order=curr_idx,
                            update_type=UpdateType.MATCHED,
                            estimated_movement=displacement
                        )
                    )
                    assigned_prev.add(prev_idx)
                    assigned_curr.add(curr_idx)

        for curr_idx in subset.curr_indices:
            if curr_idx not in assigned_curr:
                final_assignments.append(
                    MatchedStormPair(
                        prev_storm_order=None,
                        curr_storm_order=curr_idx,
                        update_type=UpdateType.NEW,
                        estimated_movement=np.array([0.0, 0.0])
                    )
                )

        return final_assignments
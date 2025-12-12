import numpy as np

from .particle_matcher import ParticleMatcher, Particle
from .storm import ShapeVectorStorm
from src.cores.base import StormsMap
from src.tracking import BaseMatcher
from dataclasses import dataclass, field
import warnings

MAX_VELOCITY = 100
MATCHING_THRESHOLD = 0.45


@dataclass
class CellSubset:
    """
    Keep track of matched storms between 2 frames. Each storm cell in source is matched with at least one storm cell in target and vice versa.
    """
    sources: set[int]      # list of indices of source storms
    targets: set[int]      # list of indices of target storms

    def add_cell(self, source: int = None, target: int = None):
        """
        add a source/target storm index to the subcell.
        """
        if source is not None:
            self.sources.add(source)
        if target is not None:
            self.targets.add(target)
    
    def merge_subcell(self, other: "CellSubset"):
        """
        merge another subcell into this one.
        """
        self.sources = self.sources.union(other.sources)
        self.targets = self.targets.union(other.targets)

    def contains(self, storm_order: int, is_source: bool=True) -> bool:
        """
        check if the cell contains the given source/target storm index.
        """
        return (storm_order in self.sources) if is_source else (storm_order in self.targets)

@dataclass
class MatchedStormPair:
    """
    Maintain matched storms pairs information.
    """
    prev_storm_order: int
    curr_storm_order: int
    displacement_list: list = field(default_factory=list)
    prev_score: float = field(default=None)
    curr_score: float = field(default=None)

    def append_displacement(self, displacement: np.ndarray):
        self.displacement_list.append(displacement)
    
    def count_matches(self) -> int:
        return len(self.displacement_list)

    def derive_motion_vector(self, dt: float) -> np.ndarray:
        if not self.displacement_list:
            warnings.warn("No displacements recorded for this matched storm pair. Returning zero vector.")
            return np.array([0.0, 0.0])
        
        return np.mean(self.displacement_list, axis=0) / dt
    
    def set_score(self, prev_score: float, curr_score: float):
        # set scores of current matching storm pair
        self.prev_score = prev_score
        self.curr_score = curr_score

class SubsetResolver:
    """
    Manage the subsets of matched storms.
    """
    matching_threshold: float

    def __init__(self, matching_threshold: float):
        self.matching_threshold = matching_threshold
    
    def create_subsets(self, assignments: list[tuple[int, int]]) -> list[CellSubset]:
        """
        create subsets of storms based on the `assignments` between 2 time frames.
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
                new_subcell = CellSubset(sources={prev_idx}, targets={curr_idx})
                subcells.append(new_subcell)
                continue
                
            # case 2: only one of them is belonged to a subcell => add the other index to that subcell
            if prev_subcell and not curr_subcell:
                prev_subcell.add_cell(target=curr_idx)
                continue
            if not prev_subcell and curr_subcell:
                curr_subcell.add_cell(source=prev_idx)
                continue
                
            # case 3: both are belonged to different subcells => merge the 2 subcells
            if prev_subcell != curr_subcell:
                prev_subcell.merge_subcell(curr_subcell)
                subcells.remove(curr_subcell)
        
        return subcells


    def resolve_subsets(
            self, subset_lst: list[CellSubset], storms_map_lst_1: StormsMap, storms_map_lst_2: StormsMap, 
            max_velocity: float, weights: list[float] = [0.5, 0.5]
        ) -> list[MatchedStormPair]:
        """
        justify the matched particles within each subset to get final assignments between 2 time frames.
        """
        # compute maximum displacement
        dt = (storms_map_lst_2.time_frame - storms_map_lst_1.time_frame).total_seconds() / 3600.0
        maximum_displacement = max_velocity * dt
        particles_matcher = ParticleMatcher()

        valid_storm_pairs: list[MatchedStormPair] = []

        for subset in subset_lst:
            prev_storms_lst: list[tuple[int, ShapeVectorStorm]] = [(idx, storm) for idx, storm in enumerate(storms_map_lst_1.storms) if idx in subset.sources]
            curr_storms_lst: list[tuple[int, ShapeVectorStorm]] = [(idx, storm) for idx, storm in enumerate(storms_map_lst_2.storms) if idx in subset.targets]

            particles_prev: list[Particle] = [Particle(feature=v, storm_order=idx) 
                                              for idx, storm in prev_storms_lst for v in storm.shape_vectors]
            particles_curr: list[Particle] = [Particle(feature=v, storm_order=idx) 
                                              for idx, storm in curr_storms_lst for v in storm.shape_vectors]

            # match particles
            particle_assignments = particles_matcher.match_particles(
                particles_prev, particles_curr, maximum_displacement, weights
            )

            matched_storms_dict = {}

            # collect matched particles information for each pair of storms
            for idx1, idx2 in particle_assignments:
                prev_storm_order = particles_prev[idx1].storm_order
                curr_storm_order = particles_curr[idx2].storm_order

                if (prev_storm_order, curr_storm_order) not in matched_storms_dict:
                    matched_storms_dict[(prev_storm_order, curr_storm_order)] = MatchedStormPair(
                        prev_storm_order=prev_storm_order,
                        curr_storm_order=curr_storm_order
                    )
                
                displacement = np.array(particles_curr[idx2].feature.coord) - np.array(particles_prev[idx1].feature.coord)
                matched_storms_dict[(prev_storm_order, curr_storm_order)].append_displacement(displacement)

            for (prev_storm_order, curr_storm_order), matched_pair in matched_storms_dict.items():
                num_particles_prev = storms_map_lst_1.storms[prev_storm_order].get_num_particles()
                num_particles_curr = storms_map_lst_2.storms[curr_storm_order].get_num_particles()
                min_particles = min(num_particles_prev, num_particles_curr)

                score = matched_pair.count_matches() / min_particles
                if score >= self.matching_threshold:
                    # score based on prev and curr storm
                    matched_pair.set_score(matched_pair.count_matches() / num_particles_prev, matched_pair.count_matches() / num_particles_curr)
                    valid_storm_pairs.append(matched_pair)
        
        return valid_storm_pairs


class StormMatcher(BaseMatcher):
    max_velocity: float
    particle_matcher: ParticleMatcher

    def __init__(self, max_velocity: float):
        self.max_velocity = max_velocity
    
    def _construct_disparity_matrix(self, object_lst1, object_lst2):
        pass

    def match_storms(
            self, storms_map_lst_1: StormsMap, storms_map_lst_2: StormsMap,
            coarse_threshold: float = 0.4, fine_threshold: float = 0.5
        ) -> list[MatchedStormPair]:
        """
        Match storms between 2 time frame.

        Args:
            storm_map1 (StormsMap): storm map in the 1st frame.
            storm_map2 (StormsMap): storm map in the 2nd frame.
        
        Returns:
            tuple[np.ndarray, list, list]:
                assignments (nd.ndarray): list of pairs of corresponding id of 2 storms.
                probability_matrix (list): list of score of corresponding assignment.
                displacements (list): list of displacement of corresponding assignment.
        """
        particles_prev: list[Particle] = [Particle(feature=v, storm_order=idx) for idx, storm in enumerate(storms_map_lst_1.storms)\
                          for v in storm.shape_vectors]
        particles_curr: list[Particle] = [Particle(feature=v, storm_order=idx) for idx, storm in enumerate(storms_map_lst_2.storms)\
                          for v in storm.shape_vectors]
        
        dt = (storms_map_lst_2.time_frame - storms_map_lst_1.time_frame).seconds / 3600
        maximum_displacement = self.max_velocity * dt
        
        if len(particles_prev) == 0 or len(particles_curr) == 0:
            return [], [], []

        self.particle_matcher = ParticleMatcher()
        particle_assignments = self.particle_matcher.match_particles(
                particles_prev, particles_curr, maximum_displacement=maximum_displacement
            )
        
        # map particles assignment back to storm.
        particles_id_prev = [p.storm_order for p in particles_prev]
        particles_id_curr = [p.storm_order for p in particles_curr]

        mapping_displacements = {curr_idx: {prev_idx: [] for prev_idx in range(len(storms_map_lst_1.storms))}\
                                 for curr_idx in range(len(storms_map_lst_2.storms))}
        
        for (p_prev_idx, p_curr_idx) in particle_assignments:
            p_prev = particles_prev[p_prev_idx].feature.coord
            p_curr = particles_curr[p_curr_idx].feature.coord
            displacement = np.array(p_curr) - np.array(p_prev)

            mapping_displacements[particles_id_curr[p_curr_idx]][particles_id_prev[p_prev_idx]].append(displacement)
        
        # count number of matched particles between each pair of storms
        matching_count = np.zeros((len(storms_map_lst_1.storms), len(storms_map_lst_2.storms)), dtype=np.int64)

        for idx1, idx2 in particle_assignments:
            matching_count[particles_prev[idx1].storm_order, particles_curr[idx2].storm_order] += 1
        
        # compute probability matrix
        ## p_A: Probability of matching where the denominator is the number of particles in the previous storm
        ## p_B: Probability of matching where the denominator is the number of particles in the curr storm
        p_A = matching_count / np.array([storm.get_num_particles() for storm in storms_map_lst_1.storms])[:, np.newaxis]
        p_B = matching_count / np.array([storm.get_num_particles() for storm in storms_map_lst_2.storms])[np.newaxis, :]

        p = np.max([p_A, p_B], axis=0)
        assignments = np.argwhere(p > coarse_threshold)

        # resolve subsets
        subset_resolver = SubsetResolver(matching_threshold=coarse_threshold)
        subsets = subset_resolver.create_subsets(assignments.tolist())
        
        valid_matched_storms = subset_resolver.resolve_subsets(
            subsets, storms_map_lst_1, storms_map_lst_2, max_velocity=self.max_velocity, weights=[0.5, 0.5]
        )

        return valid_matched_storms
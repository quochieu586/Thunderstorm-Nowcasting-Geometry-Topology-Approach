import numpy as np
import pandas as pd
from shapely.affinity import translate
from sklearn.cluster import KMeans

from .particle_matcher import ParticleMatcher, Particle
from .storm import ShapeVectorStorm
from src.cores.base import StormsMap
from src.tracking import BaseMatcher

MAX_VELOCITY = 100
MATCHING_THRESHOLD = 0.45

class StormMatcher(BaseMatcher):
    max_velocity: float
    matching_threshold: float       # minimum probability threshold for matching between 2 storms
    particle_matcher: ParticleMatcher

    def __init__(self, max_velocity: float = MAX_VELOCITY, matching_threshold: float = MATCHING_THRESHOLD):
        self.max_velocity = max_velocity
        self.matching_threshold = matching_threshold
    
    def _construct_disparity_matrix(self, object_lst1, object_lst2):
        pass

    def _resolve_displacement(
            self, displacements: list[np.ndarray], prev_storm: ShapeVectorStorm, curr_storm: ShapeVectorStorm
        ) -> np.ndarray:
        """
        Generate a final displacement from the list of particles pair displacement.
        """
        def compute_overlapping(displ: np.ndarray):
            dx, dy = displ
            pred_pol = translate(prev_storm.contour, xoff=dx, yoff=dy)

            return pred_pol.intersection(curr_storm.contour).area

        best_displacement = np.mean(displacements, axis=0)
        best_score = compute_overlapping(best_displacement)

        for num_cluster in range(2, 6):
            if len(displacements) <= num_cluster * 2:
                break
            else:
                # Build a cluster and extract the group with largest count.
                k_means = KMeans(n_clusters=num_cluster, random_state=2025)
                labels = k_means.fit_predict(displacements)

                unique_labels, counts = np.unique(labels, return_counts=True)
                largest_cluster_label = unique_labels[np.argmax(counts)]
                
                # Extract displacements in that cluster.
                optimal_displacement = k_means.cluster_centers_[largest_cluster_label]
                score = compute_overlapping(optimal_displacement)

                if score > best_score:
                    best_displacement = optimal_displacement
                    best_score = score
        
        return best_displacement

    def match_storms(
            self, storms_map1: StormsMap, storms_map2: StormsMap
        ) -> tuple[np.ndarray, list, list]:
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
        particles_prev: list[Particle] = [Particle(feature=v, storm_order=idx) for idx, storm in enumerate(storms_map1.storms)\
                          for v in storm.shape_vectors]
        particles_curr: list[Particle] = [Particle(feature=v, storm_order=idx) for idx, storm in enumerate(storms_map2.storms)\
                          for v in storm.shape_vectors]
        
        dt = (storms_map2.time_frame - storms_map1.time_frame).seconds / 3600
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

        mapping_displacements = {curr_idx: {prev_idx: [] for prev_idx in range(len(storms_map1.storms))}\
                                 for curr_idx in range(len(storms_map2.storms))}
        
        for (p_prev_idx, p_curr_idx) in particle_assignments:
            p_prev = particles_prev[p_prev_idx].feature.coord
            p_curr = particles_curr[p_curr_idx].feature.coord
            displacement = np.array(p_curr) - np.array(p_prev)

            mapping_displacements[particles_id_curr[p_curr_idx]][particles_id_prev[p_prev_idx]].append(displacement)
        
        matched_count_mapping = {k_curr: {k_prev: len(v_prev) for k_prev, v_prev in v_curr.items()} \
                                 for k_curr, v_curr in mapping_displacements.items()}

        count_df = pd.DataFrame(matched_count_mapping)

        num_particles_prev = pd.Series([s.get_num_particles() for s in storms_map1.storms], count_df.index)
        num_particles_curr = pd.Series([s.get_num_particles() for s in storms_map2.storms], count_df.columns)

        p_A = count_df.div(num_particles_prev, axis=0)
        p_B = count_df.div(num_particles_curr, axis=1)

        p = np.max([p_A, p_B], axis=0)
        assignments = np.argwhere(p > self.matching_threshold)
        return assignments, [np.array(p_B)[prev_idx][curr_idx] for prev_idx, curr_idx in assignments], \
            [self._resolve_displacement(np.array(mapping_displacements[curr_idx][prev_idx]), \
                    storms_map1.storms[prev_idx], storms_map2.storms[curr_idx]) for prev_idx, curr_idx in assignments]
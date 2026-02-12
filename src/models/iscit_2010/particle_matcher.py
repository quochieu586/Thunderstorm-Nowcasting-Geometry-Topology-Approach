import numpy as np

from dataclasses import dataclass

from src.tracking import BaseMatcher

@dataclass
class Particle:
    position: np.ndarray
    storm_order: int        # order of the storm this particle belongs to

class ParticleMatcher(BaseMatcher):
    def _construct_disparity_matrix(
            self, particle_lst1: list[Particle], particle_lst2: list[Particle], estimated_vectors: np.ndarray, 
            weights: list[float], max_displacement: float
        ):
        """
        Construct a disparity matrix for 2 lists of particles.

        Parameters:
            particle_lst1 (list[Particle]): The first list of particles.
            particle_lst2 (list[Particle]): The second list of particles.
            estimated_vectors (np.ndarray): The estimated movement vectors for each particle in `particle_lst1`.
            weights (list[float]): the list of nonnegative- and sum-to-1 weights.

        Returns:
            cost_matrix, T_D, T_S (tuple[np.ndarray, np.ndarray]): The constructed disparity matrix and distance matrix for later verification.
        """
        assert all([all([w >= 0 for w in weights]), np.sum(weights) == 1]), "weights must not be negative and must sum to 1."
        n2 = len(particle_lst2)
        coord_1 = np.array([p.position for p in particle_lst1])
        coord_2 = np.array([p.position for p in particle_lst2])

        displacement_matrix = -(coord_1[:, np.newaxis, :] - coord_2[np.newaxis, :, :])         # shape: (n1, n2, 2)
        estimated_matrix = np.repeat(estimated_vectors[:, np.newaxis, :], n2, axis=1)       # shape: (n1, n2, 2)

        # compute T_D
        dot_product = np.sum(displacement_matrix * estimated_matrix, axis=-1)  # (n1, n2)
        norms = (np.linalg.norm(displacement_matrix, axis=-1) *
                    np.linalg.norm(estimated_matrix, axis=-1) + 1e-8)
        T_D = 1 - dot_product / norms   # shape: (n1, n2)
        
        # compute T_S
        norm_displ_matrix = np.linalg.norm(displacement_matrix - estimated_matrix, axis=-1)       # shape: (n1, n2)
        max_diff_matrix = np.clip(max_displacement - norm_displ_matrix, a_min=0, a_max=max_displacement)
        T_S = 1 - (2 * np.sqrt(max_displacement * max_diff_matrix)) / (max_displacement + max_diff_matrix + 1e-6)   # shape: (n1, n2)

        # compute cost matrix
        cost_matrix = weights[0] * T_D + weights[1] * T_S

        return cost_matrix, T_D, T_S
    
    def match_particles(
            self, particle_lst1: list[Particle], particle_lst2: list[Particle], estimated_vectors: np.ndarray, 
            maximum_displacement: float, weights: list[float] = [0.5, 0.5]
        ) -> np.ndarray:
        """
        Match particles between 2 lists of particles.
        Parameters:
            particle_lst1 (list[Particle]): The first list of particles.
            particle_lst2 (list[Particle]): The second list of particles.
            maximum_displacement (float): The maximum displacement allowed for a particle to be considered a match.
            weights (list[float], default=[0.5, 0.5]): the list of nonnegative- and sum-to-1 weights.
        Returns:
            assignments (np.ndarray): The array of matched indices between the 2 lists of particles.
        """
        # print(f"ParticleMatcher weights: {weights}")
        cost_matrix, T_D, T_S = self._construct_disparity_matrix(
            particle_lst1, particle_lst2, estimated_vectors=estimated_vectors, 
            max_displacement=maximum_displacement, weights=weights
        )
        invalid_mask = (T_D > 1) | (T_S >= 1)      # violate either condition => invalid match
        
        row_ind, col_ind = self._hungarian_matching(cost_matrix)
        assignment_mask = np.zeros_like(invalid_mask, dtype=bool)
        assignment_mask[row_ind, col_ind] = True

        return np.argwhere(assignment_mask & np.logical_not(invalid_mask))
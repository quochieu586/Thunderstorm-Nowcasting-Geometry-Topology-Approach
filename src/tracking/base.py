from abc import ABC, abstractmethod
from src.cores.base import BaseObject
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

ALPHA = 2e2
EPSILON = 1e-4
MAX_ITER = 500

def solve_optimal_gamma(C_A, C_B, C_AB, P_t, P_star, alpha):
    """
    Solves for the optimal step size gamma in [0, 1] for the quadratic trace objective:
    f(gamma) = trace(C_A^T ((1 - gamma) * P_t + gamma * P_star) C_B ((1 - gamma) * P_t + gamma * P_star)^T) + alpha * trace(C_AB^T ((1 - gamma) * P_t + gamma * P_star))
    """
    delta_P = P_star - P_t
    
    # Compute the quadratic coefficient 'a'
    # a = trace(C_A * delta_P * C_B * delta_P^T)
    term_a = C_A @ delta_P @ C_B @ delta_P.T
    a = np.trace(term_a)
    
    # Compute the linear coefficient 'b'
    # b = trace(C_A * P_t * C_B * delta_P^T) + trace(C_A * delta_P * C_B * P_t^T) + alpha * trace(C_AB * delta_P^T)
    term_b1 = C_A @ P_t @ C_B @ delta_P.T
    term_b2 = C_A @ delta_P @ C_B @ P_t.T
    term_b3 = C_AB @ delta_P.T
    
    b = np.trace(term_b1) + np.trace(term_b2) + alpha * np.trace(term_b3)
    
    # Minimize f(gamma) = a * gamma^2 + b * gamma + c subject to gamma in [0, 1]
    if a > 0:
        # Convex: minimum is at the vertex, clipped to [0, 1]
        gamma_opt = -b / (2 * a)
        gamma_opt = np.clip(gamma_opt, 0.0, 1.0)
    elif a == 0:
        # Linear: slope determines minimum
        gamma_opt = 1.0 if b < 0 else 0.0
    else:
        # Concave: minimum is at the boundaries. 
        # Compare f(0) = c and f(1) = a + b + c. 
        gamma_opt = 1.0 if (a + b) < 0 else 0.0
        
    return gamma_opt

class BaseMatcher(ABC):
    """
    Base class for matching between consecutive frames.
    """
    @abstractmethod
    def _construct_disparity_matrix(
        self, object_lst1: list[BaseObject], object_lst2: list[BaseObject]
    ) -> np.ndarray:
        """
        Construct a disparity matrix for the given storm maps.

        Args:
            object_lst1 (list[BaseObject]): The first list of objects.
            object_lst2 (list[BaseObject]): The second list of objects.

        Returns:
            cost_matrix (np.ndarray): The constructed disparity matrix.
        """
        pass

    def _hungarian_matching(self, cost_matrix: np.ndarray) -> np.ndarray:
        """
        Find an optimal one-to-one assignment.

        Args:
            cost_matrix: (np.ndarray): The constructed disparity matrix.

        Returns:
            assignments (tuple[np.ndarray]): The list of x-, y- assignments.
        """
        return linear_sum_assignment(cost_matrix)

    def _quadratic_assignment(self, internal_cost_1: np.ndarray, internal_cost_2: np.ndarray, 
                              cost_matrix: np.ndarray, alpha=ALPHA):
        """
        Matching with quadratic assignment problem (QAP) formulation, solved approximately via entropic fused Gromov-Wasserstein in POT. Where:
            - internal_cost_1: pairwise cost matrix for the first set of objects (e.g., distance between storms in the first frame)
            - internal_cost_2: pairwise cost matrix for the second set of objects (e.g., distance between storms in the second frame)
            - cost_matrix: unary cost matrix between objects in the first and second set (e.g., distance between storm centroids across frames)
        Objective function:
            min_P (trace(C_A^T P C_B P^T)) + alpha * trace(C_AB^T P)
        """
        m, n = internal_cost_1.shape[0], internal_cost_2.shape[0]
        N = max(m, n)
        
        # 1. Dummy Padding to make matrices square N x N
        C_A = np.zeros((N, N))
        C_A[:m, :m] = internal_cost_1

        C_B = np.zeros((N, N))
        C_B[:n, :n] = internal_cost_2

        C_AB = np.zeros((N, N))
        C_AB[:m, :n] = cost_matrix

        # 2. Initialize P as a flat, doubly stochastic matrix
        P = np.ones((N, N)) / N
        
        # 3. Frank-Wolfe Optimization Loop
        for i in range(MAX_ITER):
            # Compute Gradient grad(P)
            grad = 2 * (C_A @ P @ C_B.T) + alpha * C_AB
            row_ind, col_ind = linear_sum_assignment(grad)

            # P_dir: P_direction is the update direction
            P_dir = np.zeros((N, N))
            P_dir[row_ind, col_ind] = 1.0
            
            # Compute the optimal step size
            gamma = solve_optimal_gamma(C_A, C_B, C_AB, P, P_dir, alpha)
            P = (1.0 - gamma) * P + gamma * P_dir

            # Check for convergence (optional)
            if np.linalg.norm(P - P_dir, 'fro') < EPSILON:
                break

        # 4. Final Projection to a strict Discrete Permutation Matrix
        row_ind, col_ind = linear_sum_assignment(-P)
        P_discrete = np.zeros((N, N))
        P_discrete[row_ind, col_ind] = 1.0
        
        # 5. Crop back to the original unbalanced m x n shape
        P_final = P_discrete[:m, :n]
        row_ind_final, col_ind_final = np.where(P_final == 1)
        return row_ind_final, col_ind_final

class BaseTrackingHistory(ABC):
    """
    Track the history of storms.
    """
    tracks: list[dict]  # list of track
    storm_dict: dict    # key -> id of current storm, val -> id of corresponding track
    active_list = list  # keep the id of current active storms

class BaseTracker(ABC):
    """
    Track storms over time.
    """
    matcher: BaseMatcher
    tracker: BaseTrackingHistory
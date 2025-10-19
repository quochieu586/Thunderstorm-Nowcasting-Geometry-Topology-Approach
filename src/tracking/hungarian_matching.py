import numpy as np

def _row_reduce(cost_matrix: np.ndarray) -> np.ndarray:
    """
    Perform row reduction on the cost matrix.

    Args:
        cost_matrix (np.ndarray): The cost matrix to be reduced.

    Returns:
        reduced_matrix (np.ndarray): The row-reduced cost matrix.
    """
    reduced_matrix = cost_matrix - cost_matrix.min(axis=1, keepdims=True)
    return reduced_matrix

def _col_reduce(cost_matrix: np.ndarray) -> np.ndarray:
    """
    Perform column reduction on the cost matrix.

    Args:
        cost_matrix (np.ndarray): The cost matrix to be reduced.

    Returns:
        reduced_matrix (np.ndarray): The column-reduced cost matrix.
    """
    reduced_matrix = cost_matrix - cost_matrix.min(axis=0, keepdims=True)
    return reduced_matrix

def _extract_soft_assignments(cost_matrix: np.ndarray, row_major: bool=True, total: float=1e-8) -> list[tuple[int, int]]:
    """
    Perform the Hungarian soft assignment version algorithm to find optimal assignments in the cost matrix.

    Args:
        cost_matrix (np.ndarray): The cost matrix.
        row_major (bool): Whether to perform row reduction first.
        total (float): A small value to tolerate the near-zero assignment.
    Returns:
        assignments (list[tuple[int, int]]): The list of assignments as (row, column) tuples.
    """
    M = cost_matrix.copy()
    if row_major:
        M = _row_reduce(M)
        M = _col_reduce(M)
    else:
        M = _col_reduce(M)
        M = _row_reduce(M)
    
    return np.argwhere(M <= total).tolist()

def reduced_soft_hungarian(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """
    Perform the Hungarian soft assignment version algorithm to find optimal assignments in the cost matrix.

    Args:
        cost_matrix (np.ndarray): The cost matrix.
    Returns:
        assignments (list[tuple[int, int]]): The list of assignments as (row, column) tuples.
    """
    row_assignments = set([tuple(assignment) for assignment in _extract_soft_assignments(cost_matrix, row_major=True)])
    col_assignments = set([tuple(assignment) for assignment in _extract_soft_assignments(cost_matrix, row_major=False)])

    return list(row_assignments.union(col_assignments))

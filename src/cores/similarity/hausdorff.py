from scipy.spatial.distance import directed_hausdorff
from src.cores.contours import StormObject

def hausdorff_distance(storm1: StormObject, storm2: StormObject) -> float:
    """
    Compute the Hausdorff distance between two storm objects.
    """
    c1 = storm1.contour.reshape(-1, 2)
    c2 = storm2.contour.reshape(-1, 2)

    d1 = directed_hausdorff(c1, c2)[0]
    d2 = directed_hausdorff(c2, c1)[0]

    return max(d1, d2)
from src.cores.base import StormObject
from shapely import hausdorff_distance

def compute_hausdorff_distance(storm1: StormObject, storm2: StormObject) -> float:
    """
    Compute the Hausdorff distance between two storm objects.
    """
    d1 = hausdorff_distance(storm1.contour, storm2.contour)
    d2 = hausdorff_distance(storm2.contour, storm1.contour)
    
    return max(d1, d2)
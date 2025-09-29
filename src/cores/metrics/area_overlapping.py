import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from src.cores.contours import StormsMap

def overlapping_storm_area(contour_1: Polygon, contour_2: Polygon) -> float:
    """
    Calculate the overlapping area between two contours.
    """
    intersection = contour_1.intersection(contour_2)
    union = contour_1.union(contour_2)
    return intersection.area / union.area if union.area > 0 else 0.0

def overlapping_area(storm1: StormsMap, storm2: StormsMap) -> float:
    """
    Calculate the overlapping area between two StormsMap objects.

    Parameters:
    storm1 (StormsMap): The first storm object.
    storm2 (StormsMap): The second storm object.

    Returns:
    - Overlapping: The area of overlap between the two storms.
    - Precision: The ratio of the overlapping area to the area of storm1.
    - Recall: The ratio of the overlapping area to the area of storm2.
    """
    return 

def recall(pred_map: StormsMap, true_map: StormsMap) -> float:
    """
    Percentage of actual storms area that model can predict correctly.
    """
    overlapping = overlapping_area(pred_map, true_map)
    true_area = sum(cv2.contourArea(np.array(storm.contour)) for storm in true_map.storms)
    
    return overlapping / true_area if true_area > 0 else 0.0

def precision(pred_map: StormsMap, true_map: StormsMap) -> float:
    """
    Percentage of predicted storms area that is actually correct.
    """
    overlapping = overlapping_area(pred_map, true_map)
    pred_area = sum(cv2.contourArea(np.array(storm.contour)) for storm in pred_map.storms)
    
    return overlapping / pred_area if pred_area > 0 else 0.0

def f1_score(pred_map: StormsMap, true_map: StormsMap) -> float:
    """
    Harmonic mean of precision and recall for storms prediction.
    """
    p = precision(pred_map, true_map)
    r = recall(pred_map, true_map)
    
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
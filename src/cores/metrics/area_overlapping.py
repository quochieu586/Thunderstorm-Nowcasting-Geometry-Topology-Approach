import cv2
import numpy as np

from src.cores.contours import StormsMap

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
    assert storm1.map_size == storm2.map_size, "Storm maps must be of the same size"

    # Assuming StormsMap has a method to get the area as a set of coordinates
    map1 = np.zeros(storm1.map_size, dtype=np.uint8)
    map2 = np.zeros(storm2.map_size, dtype=np.uint8)

    for storm in storm1.storms:
        cv2.fillPoly(map1, [np.array(storm.contour)], 1)

    for storm in storm2.storms:
        cv2.fillPoly(map2, [np.array(storm.contour)], 1)

    overlapping_area = cv2.bitwise_and(map1, map2)
    return np.sum(overlapping_area)  # or calculate actual area if coordinates are in a specific format

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
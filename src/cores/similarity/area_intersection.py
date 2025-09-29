import cv2
import numpy as np

from src.cores.contours import StormObject

def area_intersection(storm1: StormObject, storm2: StormObject) -> float:
    """
        Calculate the area of intersection between two storm objects.

        Args:
            storm1 (StormObject): The first storm object.
            storm2 (StormObject): The second storm object.

        Returns:
            float: The area of intersection between the two storm objects.
    """
    # Create blank images to draw the contours
    img1 = np.zeros((1000, 1000), dtype=np.uint8)
    img2 = np.zeros((1000, 1000), dtype=np.uint8)

    # Draw the contours on the blank images
    cv2.drawContours(img1, [storm1.contour], -1, color=255, thickness=-1)
    cv2.drawContours(img2, [storm2.contour], -1, color=255, thickness=-1)

    # Calculate the intersection
    intersection = cv2.bitwise_and(img1, img2)

    # Calculate the area of the intersection
    area = cv2.countNonZero(intersection)

    return area
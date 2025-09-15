import numpy as np
import cv2
from typing import List, Tuple
import numpy as np


def _filter_words(image: np.ndarray) -> np.ndarray:
    """
        Filter words
    """
    mask = np.where(np.sum(image, axis=-1) >= 550, 1, 0).astype(np.uint8)
    kernel = np.ones((3,3), dtype=np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=3)

    return cv2.inpaint(image.astype(np.uint8), dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def _filter_foreground(image: np.ndarray) -> np.ndarray:
    """
        Filter those country boundaries inside the weather regions.
    """
    sharpen_kernel = np.array([
                    [-1/2, -1/2, -1/2],
                    [-1/2,    5, -1/2],
                    [-1/2, -1/2, -1/2]
                ])

    filter_img = cv2.filter2D(image.astype(np.int16), -1, sharpen_kernel)

    diff_kernel = np.array([
                    [-1/2, -1/2, -1/2],
                    [-1/2,    4, -1/2],
                    [-1/2, -1/2, -1/2]
                ])

    filter_img = cv2.filter2D(filter_img, -1, diff_kernel)

    mask = np.all(filter_img < -50, axis=-1).astype(np.uint8) * 255
    kernel = np.ones((3,3), dtype=np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=3)

    return cv2.inpaint(image, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA).astype(np.uint8)

def _filter_background(image: np.ndarray) -> np.ndarray:
    """
        Filter background
    """
    # Calculate variance across the color channels for each pixel in a vectorized way
    pixel_var = np.var(image, axis=-1)
    
    # Create a mask: Replace background pixels with 255, keep original pixels for non-background
    mask = np.where(pixel_var <= 50, 0, 1)
    mask_3d = np.stack([mask] * 3, axis=-1)
    background_converter = 1 - mask_3d

    return image * mask_3d + background_converter * 144

def _preprocess(image: np.ndarray) -> np.ndarray:
    image = _filter_background(_filter_words(image)).astype(np.uint8)
    return _filter_foreground(image)

def _convert_to_dbz(image: np.ndarray, arr_colors: np.ndarray):
    k = len(arr_colors)
    img_reshape = image.reshape(-1,3)
    arr = np.zeros(shape=[img_reshape.shape[0], k-1, 2])

    for i in range(k-1):
        color1, dbz1 = arr_colors[i]
        color2, dbz2 = arr_colors[i + 1]
        
        # Compute Euclidean distances from the pixel to color1 and color2
        dist1 = np.linalg.norm(img_reshape - color1, axis=1)
        dist2 = np.linalg.norm(img_reshape - color2, axis=1)

        arr[:, i, 0] = dist1 + dist2
        arr[:, i, 1] = (dbz1 * dist2 + dbz2 * dist1) / (dist1 + dist2)
    
    distance_values = arr[:, :, 0]
    min_distance_interval = np.argmin(distance_values, axis=1)
    result = arr[np.arange(arr.shape[0]), min_distance_interval]

    return result[:,1].reshape(image.shape[:2])

def extract_contour_by_dbz(img: np.ndarray, thresholds: List[int], sorted_color: list[Tuple[Tuple[int, int, int], int]]):
    """
        Draw the DBZ contour for the image.

        Args:
            img: source image.
            thresholds: dbz thresholds for drawing contours.
            sorted_color: a list of tuples where each tuple includes:
                - a color represented as an RGB triplet (e.g., (R, G, B)).
                - a corresponding dBZ value, sorted in increasing order of dBZ.
        
        Returns:
            Tuple[np.ndarray,List[np.ndarray]]: A tuple containing:
                - contour_img (np.ndarray): A blank image with the extracted contours drawn.
                - contours (List[np.ndarray]): A list of detected contours, each represented as an array of points.
                - contour_colors (List(Tuple[int,int,int])): A list of color corresponding with each contours
    """
    img = _preprocess(img)

    dbz_map = _convert_to_dbz(img, sorted_color).astype(np.uint8)
    
    blank_img = np.ones(shape=img.shape, dtype=np.uint8) * 255

    # Get the region
    region = np.digitize(dbz_map, bins=thresholds).astype(np.uint8)      # 99.9: background
    contours = []
    contour_colors = []

    # Draw the contour
    for idx in range(len(thresholds)):
        color_layer = ((region >= idx+1) & (region <= len(thresholds))).astype(np.uint8)

        mean_color = img[region == idx + 1].mean(axis=0)
        contour, _ = cv2.findContours(color_layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours.append(contour)
        contour_colors.append(mean_color)
        cv2.drawContours(blank_img, contour, -1, mean_color, 2)

    return blank_img, contours, contour_colors


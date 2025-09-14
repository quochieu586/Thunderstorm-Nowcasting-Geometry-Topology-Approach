from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import cv2
import math
from dataclasses import dataclass

def _simplify_contour(contour: np.ndarray) -> np.ndarray:
    """
        Reduce the vertices of contour, using Douglas-Peucker algorithm.
        
        Epsilon is chosen as:
            epsilon = log(Area of contour)

        Args:
            contour (np.ndarray): the contour of storm.
        
        Returns:
            simplified_contour (np.ndarray): the simplified contour.
    """
    epsilon = math.log(cv2.contourArea(contour))
    output_contour = cv2.approxPolyDP(contour, epsilon, True)
    return output_contour if len(output_contour) >= 3 else contour

def process_contours(contours: List[np.ndarray], area_threshold: int = 15) -> List[np.ndarray]:
    """
        Process the contours:
            1. Drop the contour that its area is too small.
            2. Simplify the contour.
            3. Convert contours into polygons, fix case that polygon is not valid.
            4. Sort  based on the area.

        Args:
            contours (List[np.ndarray]): a list of contours.
            area_threshold (int, optional): the minimum area allowance, unit: pixel. Default is 15.

        Returns:
            processed_polygons (List[Polygons]): a list of processed polygons.
    """

    processed_contours = [_simplify_contour(contour) for contour in contours \
                          if cv2.contourArea(contour) >= area_threshold]
    processed_polygons = contours_to_polygons(processed_contours)
    return sorted(processed_polygons, key= lambda x: x.area, reverse=True)

def contours_to_polygons(contours: List[List[np.ndarray]]) -> List[Polygon]:
    """
        Convert the list of contours into the list of polygons.
    """
    if isinstance(contours[0], List):
        contours = [contour for subcontours in contours for contour in subcontours]
    
    polygons = []
    for contour in contours:
        polygon = Polygon(contour.squeeze(axis=1))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.geom_type == "MultiPolygon":
            polygons.extend(list(polygon.geoms))
        else:
            polygons.append(polygon)
    
    return polygons

def polygons_to_contours(polygons: List[Polygon]) -> List[np.ndarray]:
    """
        Convert a list  shapely Polygon back into a list of numpy contour format (N, 1, 2).
    """
    contours = []
    for polygon in polygons:
        if polygon.is_empty:
            continue

        # Take exterior coords (skip last point since shapely closes it automatically)
        coords = np.array(polygon.exterior.coords[:-1], dtype=np.int32)
        
        # Reshape into OpenCV contour format
        contours.append(coords.reshape(-1, 1, 2))

    return contours

def construct_shape_vector(polygons: List[Polygon], point: Tuple[float, float], radii = [20, 40, 60], num_sectors = 8):
    """
        Construct the shape vector of polygons around a point.

        Args:
            polygons (List[Polygon]): a list of polygons.
            point: (x, y) tuple (center point A).
            radii: a list of radius of sectors (default: [20, 40, 60]).
            num_sectos: number of sectors (default: 8).

        Returns:
            features_vector: a feature vector of radii x num_sectors shape.
    """
    poly_union = unary_union(polygons) if len(polygons) > 1 else polygons[0]
    A = Point(point)

    features_vector = []
    for r in radii:
        for j in range(num_sectors):
            angle_start = j * (360 / num_sectors)
            angle_end = (j + 1) * (360 / num_sectors)

            sector = construct_sector(A, radius=r, angle_start=angle_start, angle_end=angle_end)

            features_vector.append(poly_union.intersection(sector).area)
    
    for i in range(len(features_vector)):
        div = i // num_sectors
        rem = i % num_sectors
        for j in range(div):
            features_vector[i] -= features_vector[num_sectors * j + rem]
    
    return np.array(features_vector)

def construct_sector(center, radius, angle_start, angle_end, num_points=30):
    """
        Make a sector (wedge) polygon.
        
        Args:
            center (tuple): The position of center point.
            radius (float): The radius of sector.
            angle_start (float): The degree value of start angle
            angle_end (float): The degree value of end angle
            num_points (int, optional): The number of points for approximating the
                sector arc. Default to 30.
        
        Returns:
            polygon (Polygon): The approximated polygon of the sector.
    """
    cx, cy = center.x, center.y
    angles = np.linspace(np.deg2rad(angle_start), np.deg2rad(angle_end), num_points)
    arc = [(cx + radius*np.cos(a), cy + radius*np.sin(a)) for a in angles]
    return Polygon([center.coords[0]] + arc + [center.coords[0]])

def _compute_overlapping_pair(pol_1: Polygon, pol_2: Polygon) -> Tuple[float, float]:
    """
        Compute overlapping between 2 polygons pol1 and pol2.

        Args:
            pol_1 (Polygon): First polygon.
            pol_2 (Polygon): Second polygon.

        Returns:
            Tuple[float, float]: Overlapping ratio between pol1 and pol2.
                The first value is the ratio of the area of pol1 that is overlapped by pol2.
                The second value is the ratio of the area of pol2 that is overlapped by pol1.
    """
    area_1 = pol_1.area
    area_2 = pol_2.area
    common_area = pol_1.intersection(pol_2).area

    return common_area / area_1, common_area / area_2

def compute_overlapping(
        polygons_1: Polygon, 
        polygons_2: Polygon
    ) -> np.ndarray:
    """
        Compute overlapping between two lists of polygons.

        Args:
            polygons_1 (Polygon): List of polygons in the first frame.
            polygons_2 (Polygon): List of polygons in the second frame.

        Returns:
            np.ndarray: Overlapping matrix of shape (len(contours_1), len(contours_2)).
                The value at position (i, j) is the overlapping ratio between contours_1[i] and contours_2[j].
                The value is the ratio of the area of contours_1[i] that is overlapped by contours_2[j].
    """
    overlapping_matrix = np.zeros((len(polygons_1), len(polygons_2)), dtype=np.float32)

    for i, pol_1 in enumerate(polygons_1):
        for j, pol_2 in enumerate(polygons_2):
            overlapping_matrix[i, j], _ = _compute_overlapping_pair(pol_1, pol_2)

    return overlapping_matrix

@dataclass
class ShapeVector:
    coord: Tuple[float, float]
    vector: np.ndarray

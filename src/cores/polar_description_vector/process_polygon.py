from typing import List, Tuple, Union
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import cv2
import math
from dataclasses import dataclass
from shapely.affinity import rotate, translate

from src.preprocessing import convert_contours_to_polygons, convert_polygons_to_contours

def simplify_contour(contour: np.ndarray) -> np.ndarray:
    """
        Reduce the vertices of contour, using Douglas-Peucker algorithm.

        Args:
            contour (np.ndarray): the contour of storm.
        
        Returns:
            simplified_contour (np.ndarray): the simplified contour.
    """
    epsilon = math.log(cv2.contourArea(contour))
    output_contour = cv2.approxPolyDP(contour, epsilon, True)
    return output_contour if len(output_contour) >= 3 else contour

def construct_shape_vector(polygons: List[Union[np.ndarray, Polygon]], point: Tuple[float, float], radii = [20, 40, 60], num_sectors = 8):
    """
        Construct the shape vector of polygons around a point.

        Args:
            polygons (List[np.ndarray]): a list of contours of polygons.
            point: (x, y) tuple (center point A).
            radii: a list of radius of sectors (default: [20, 40, 60]).
            num_sectors: number of sectors (default: 8).

        Returns:
            features_vector: a feature vector of radii x num_sectors shape.
    """
    if isinstance(polygons[0], np.ndarray):
        polygons: list[Polygon] = convert_contours_to_polygons(polygons)

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


def construct_shape_vector_fast(global_contours: list[np.ndarray], particles: np.ndarray, num_sectors: int, radii: list[float]) -> np.ndarray:
    """
    Construct the shape vector for a list of particles.

    Args:
        global_contours (list[np.ndarray]): the list of contours representing the global storm area.
        particles (np.ndarray): the list of particles, shape (num_particles, 2).
    
    Returns:
        shape_vectors (np.ndarray): the shape vectors, shape (num_particles, num_radii * num_sectors).
    """
    def _precompute_sector_templates(r: float, num_points: int = 15):
        base_sector = construct_sector((0,0), r, 0, 360/num_sectors, num_points=num_points)
        points_sector_templates = []
        for j in range(num_sectors):
            sector = rotate(base_sector, j*360/num_sectors, origin=(0,0))
            points_sector_templates.append(np.array(sector.exterior.coords))
        return points_sector_templates

    shape_vectors = np.zeros(shape=(len(radii), len(particles), num_sectors))
    global_poly = unary_union(convert_contours_to_polygons(global_contours))

    for r_idx, r in enumerate(radii):
        points_sector_templates = _precompute_sector_templates(r=r, num_points=15)
        sector_polys = [Polygon(s) for s in points_sector_templates]

        for i, (x, y) in enumerate(particles):
            for j, sp in enumerate(sector_polys):
                shifted = translate(sp, xoff=x, yoff=y)
                shape_vectors[r_idx, i, j] = shifted.intersection(global_poly).area

    actual_shape_vectors = np.zeros_like(shape_vectors)
    actual_shape_vectors[0] = shape_vectors[0]
    for r_idx in range(1, len(radii)):
        actual_shape_vectors[r_idx] = shape_vectors[r_idx] - shape_vectors[r_idx - 1]

    # reshape into (num_particles, num_radii * num_sectors)
    return np.concatenate(actual_shape_vectors, axis=-1)


def construct_sector(center: np.ndarray, radius: float, angle_start: float, angle_end: float, num_points: int = 30) -> np.ndarray:
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
            polygon (np.ndarray): The approximated polygon of the sector.
    """
    center = Point(center)

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

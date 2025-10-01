import numpy as np
from dataclasses import dataclass
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

Coord = tuple[float, float]

@dataclass
class PointShapeVector:
    coord: tuple[float, float]
    vector: np.ndarray

def contours_to_polygons(contours: list[list[np.ndarray]]) -> list[Polygon]:
    """
        Convert the list of contours into the list of polygons.
    """
    if isinstance(contours[0], list):
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

def polygons_to_contours(polygons: list[Polygon]) -> list[np.ndarray]:
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

def construct_shape_vector(polygons: list[Polygon], point: tuple[float, float], radii = [20, 40, 60], num_sectors = 8):
    """
        Construct the shape vector of polygons around a point.

        Args:
            polygons (list[Polygon]): a list of polygons.
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
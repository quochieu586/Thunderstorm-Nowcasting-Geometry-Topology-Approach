import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import cv2

def construct_shape_vector(polygons, vertex, radii = [20, 40, 60], num_sectors = 8):
    """
    Args:
        polygons: a list of polygons where each is represented by a list of [x,y] (vertex).
        vertex: (x, y) tuple (center point A).
        radii: a list of radius of sectors (default: [20, 40, 60]).
        num_sectos: number of sectors (default: 8).

    Returns:
        features_vector: a feature vector of radii $\times$ num_sectors shape.
    """
    polygons = [Polygon(np.squeeze(pol, axis=1)) for pol in polygons]
    poly_union = unary_union(polygons) if len(polygons) > 1 else polygons[0]
    A = Point(vertex)

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
    
    return features_vector

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

def plot_contour(contour, shape, ax = None, figsize=(6, 6), color = (255, 0, 0), thickness = 2, new_fig = True, blank_img = None):
    """
    Plot a contour on a blank or existing image using OpenCV and Matplotlib.

    Args:
        contour (np.ndarray): The contour to plot, typically obtained from 
            `cv2.findContours`. Expected shape is (N, 1, 2).
        shape (tuple): Shape of the output image (H, W) for grayscale or 
            (H, W, C) for color.
        ax (matplotlib.axes.Axes, optional): If provided, the image will be 
            drawn on this Matplotlib axis. Defaults to None.
        figsize (tuple, optional): Size of the new figure if `new_fig` is True. 
            Defaults to (6, 6).
        color (tuple, optional): Color of the contour in BGR format (as OpenCV expects).
            Defaults to (255, 0, 0).
        thickness (int, optional): Thickness of the contour line. Defaults to 2.
        new_fig (bool, optional): Whether to create a new figure. Defaults to True.
        blank_img (np.ndarray, optional): Existing image to draw on. If None, 
            a white blank image of given `shape` is created. Defaults to None.
    """
    if new_fig:
        plt.figure(figsize=figsize)

    if blank_img is None:
        blank_img = np.ones(shape=shape, dtype=np.int16) * 255
    cv2.drawContours(blank_img, [contour], contourIdx=-1, color=color, thickness=thickness)

    if ax is not None:
        ax.imshow(blank_img)

    else:
        plt.imshow(blank_img)
        if new_fig:
            plt.show()
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_shape_vector(
        contours: list[np.ndarray], p: tuple[float, float], img_shape: np.ndarray, radii: float, num_sectors: list[float], figsize=(7,7)
    ):
    """
    Visualize the shape vector in the image.
    """
    plt.figure(figsize=figsize)
    fig = plt.gcf()
    ax = fig.gca()

    blank_img = np.ones(img_shape, dtype=np.int16) * 255

    # Plot polygon (assuming you have a helper function plot_contour)
    cv2.drawContours(blank_img, contours, -1, color=(200,0,0), thickness=1)

    # Add circles
    for r in radii:
        circle = plt.Circle(p, radius=r, color='green', fill=False, linestyle="--")
        ax.add_patch(circle)

    # Add radial lines (sectors)
    longest_r = max(radii)
    for i in range(num_sectors):
        grad_angle = np.deg2rad(i * (360 / num_sectors))
        x_start, y_start = p
        x_end = x_start + np.cos(grad_angle) * longest_r
        y_end = y_start + np.sin(grad_angle) * longest_r
        plt.plot([x_start, x_end], [y_start, y_end], color="green", linewidth=1)

    # Mark the reference point
    plt.scatter([p[0]], [p[1]], marker='o', s=30, color="blue", label="Point A")

    plt.imshow(blank_img)
    plt.show()
import cv2
import numpy as np
from typing import Tuple, Any, Union
import matplotlib.pyplot as plt

def read_image(path: str):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_orthogonal_hull(orthogonal_hull: Union[list, Any], image: np.ndarray, color: Tuple[int, int, int] = (0,0,255)):

    for i in range(len(orthogonal_hull) - 1):
        cv2.line(image, orthogonal_hull[i], orthogonal_hull[i + 1], color, 2)
    # Vẽ đường nối từ điểm cuối về điểm đầu
    cv2.line(image, orthogonal_hull[-1], orthogonal_hull[0], color, 2)

    return image

def write_image(path: str, img: np.ndarray):
    """
        Write image to destination path.
    """
    write_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, write_img)

def plot_contour(contour, shape, figsize=(6, 6), color = (255, 0, 0), thickness = 2):
    plt.figure(figsize=figsize)

    blank_img = np.ones(shape=shape, dtype=np.int16) * 255

    cv2.drawContours(blank_img, [contour], contourIdx=-1, color=color, thickness=thickness)

    plt.imshow(blank_img)
    plt.show()
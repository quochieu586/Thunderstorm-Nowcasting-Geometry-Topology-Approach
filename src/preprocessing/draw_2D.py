import cv2
import numpy as np
import xarray as xr
from typing import Tuple, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
import pyart

def read_image(path: str):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_grib(path: Path):
    ds = xr.open_dataset(path, engine='cfgrib',backend_kwargs={"decode_timedelta": True})
    # print(ds)  # Show dataset metadata

    # Pick the first data variable (e.g., reflectivity)
    var = list(ds.data_vars)[0]
    data = ds[var]
    arr = data.squeeze().values
    return arr

def read_nexrad_radar(path: Path):
    """
    Data files: KLIX20210829_050151_V06.ar2v
    Segment | Meaning
    KLIX    | Radar station ID
    20210829| Date (YYYYMMDD) August 29, 2021
    050151  | Time (HHMMSS) 05:01:51 UTC
    V06     | Volume number (scan sequence) Volume 6 of that radar cycle
    """

    radar = pyart.io.read_nexrad_archive(path)
    print("Fields:", radar.fields.keys())
    print("Elevation angles:", radar.fixed_angle['data'])
    print("Scan start:", radar.time['units'])
    print("Latitude:", radar.latitude['data'][0])

    grid = pyart.map.grid_from_radars(
        radar,
        grid_shape=(1, 400, 400),  # Higher grid size = more coverage
        grid_limits=((0, 1000), (-400000, 400000), (-400000, 400000)),  # ±400 km
        fields=['reflectivity'],
        weighting_function='nearest',
        gridding_algo='map_gates_to_grid'
    )

    ref = grid.fields['reflectivity']['data'].filled(np.nan)[0]  # 2D NumPy array

    from pyart.util import datetimes_from_radar
    time_datetimes = datetimes_from_radar(radar)  # list of datetime objects
    scan_start_time = time_datetimes[0]           # first time (start of scan)
    # scan_end_time = time_datetimes[-1]            # last time (end of scan)
    return ref, scan_start_time

def read_nexrad_grid(path: Path):
    grid = pyart.io.read_grid(path)
    data = grid.fields['reflectivity']['data']

    # Retrieve the composed reflectivity by taking the maximum value along the vertical axis
    composite_refl_2d = np.max(data, axis=0)
    fill_value = grid.fields['reflectivity']['_FillValue']
    composite_refl_2d = np.ma.masked_where(composite_refl_2d == fill_value, composite_refl_2d)
    return composite_refl_2d, grid.time


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
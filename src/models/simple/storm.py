from shapely.geometry import Polygon
from typing import Union
import numpy as np

from src.cores.base.contours import StormObject
from src.preprocessing import convert_polygons_to_contours
from src.cores.polar_description_vector import ShapeVector, simplify_contour, construct_shape_vector

class StormShapeVectors(StormObject):
    shape_vectors: list[ShapeVector]
    coords: np.ndarray

    def __init__(self, contour: Union[Polygon, np.ndarray], id: str = ""):
        super().__init__(contour=contour, id=id)

        contour_arr = convert_polygons_to_contours([contour])[0] if isinstance(contour, Polygon) else contour
        self.shape_vectors = []

        try:
            self.coords = simplify_contour(contour_arr)
        except Exception as e:
            print(f"Error when working with contour {contour_arr.shape}: {contour_arr}")
            print(f"Contour: {self.contour} | Area: {self.contour.area}")
            raise e
        

    def extract_shape_vectors(self, global_contours: list[Polygon], radii = [20, 40, 60], num_sectors = 8):
        vectors = [construct_shape_vector(polygons=global_contours, point=coord, radii=radii, num_sectors=num_sectors) for coord in self.coords]
        self.shape_vectors = [ShapeVector(coord=(coord[0], coord[1]), vector=vector) for coord, vector in zip(self.coords.reshape(-1, 2), vectors)]
        return self.shape_vectors
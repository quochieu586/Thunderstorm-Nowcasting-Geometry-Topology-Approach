import os
from src.identification import (
    SimpleContourIdentifier, HypothesisIdentifier,
    MorphContourIdentifier, ClusterIdentifier, BaseStormIdentifier
)

BASE_FOLDER = "data/images"
IMAGES_FOLDER = os.listdir(os.path.join(BASE_FOLDER))

def get_all_images(folder: str):
    """
    Get all images in files
    """
    images = [file for file in os.listdir(os.path.join(BASE_FOLDER, folder)) if file.endswith('.png') or file.endswith('.jpg')]
    images.sort(reverse=False)
    
    return images

IDENTIFICATION_METHODS: dict[str, BaseStormIdentifier] = {
    "Simple Contour": SimpleContourIdentifier(threshold=30, filter_area=50),
    "Hypothesis": HypothesisIdentifier(threshold=30, filter_area=50, distance_dbz_threshold=5, filter_center=10),
    "Morphology": MorphContourIdentifier(threshold=30, n_thresh=3, filter_area=50, center_filter=10),
    "Cluster": ClusterIdentifier(thresholds=[30, 35, 40, 45, 50], filter_area=50, filter_center=10)
}
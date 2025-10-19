import os
from src.identification import (
    SimpleContourIdentifier, HypothesisIdentifier,
    MorphContourIdentifier, ClusterIdentifier
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

IDENTIFICATION_METHODS = {
    "Simple Contour": SimpleContourIdentifier(),
    "Hypothesis": HypothesisIdentifier(),
    "Morphology": MorphContourIdentifier(),
    "Cluster": ClusterIdentifier()
}
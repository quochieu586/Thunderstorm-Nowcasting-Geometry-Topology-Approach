import os
from src.identification import (
    SimpleContourIdentifier, HypothesisIdentifier,
    MorphContourIdentifier, ClusterIdentifier, BaseStormIdentifier
)

from src.models import (
    BasePrecipitationModeling, SimplePrecipitationModel, ETitanPrecipitationModel
)

BASE_FOLDER = "data/images"
IMAGES_FOLDER = os.listdir(os.path.join(BASE_FOLDER))

MODELS = dict[str]

IDENTIFICATION_METHODS: dict[str, BaseStormIdentifier] = {
    "Simple Contour": SimpleContourIdentifier(),
    "Hypothesis": HypothesisIdentifier(distance_dbz_threshold=5, filter_center=10),
    "Morphology": MorphContourIdentifier(n_thresh=3, center_filter=10),
    "Cluster": ClusterIdentifier(filter_center=10)
}

PRECIPITATION_MODELS: dict[str, BasePrecipitationModeling] = {
    "Simple Precipitation Model": SimplePrecipitationModel(SimpleContourIdentifier()),
    "ETitan Precipitation Model": ETitanPrecipitationModel(MorphContourIdentifier(n_thresh=3, center_filter=10))
}
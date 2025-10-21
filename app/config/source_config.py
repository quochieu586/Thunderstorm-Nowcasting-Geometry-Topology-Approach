import os
from src.identification import (
    SimpleContourIdentifier, HypothesisIdentifier,
    MorphContourIdentifier, ClusterIdentifier, BaseStormIdentifier
)

from src.models import (
    BasePrecipitationModel, SimplePrecipitationModel, ETitanPrecipitationModel, 
    TitanPrecipitationModel, OursPrecipitationModel, AdaptiveTrackingPrecipitationModel
)

THRESHOLD_OPTIONS = [20, 25, 30, 35, 40, 45, 50]
FILTER_AREA_OPTIONS = [5, 10, 15, 20, 25, 30, 40, 50]

BASE_FOLDER = "data/images"

IDENTIFICATION_METHODS: dict[str, BaseStormIdentifier] = {
    "Simple Contour": SimpleContourIdentifier(),
    "Hypothesis": HypothesisIdentifier(distance_dbz_threshold=5, filter_center=10),
    "Morphology": MorphContourIdentifier(n_thresh=3, center_filter=10),
    "Cluster": ClusterIdentifier(filter_center=10)
}

PRECIPITATION_MODELS: dict[str, BasePrecipitationModel] = {
    "Simple Precipitation Model": SimplePrecipitationModel(SimpleContourIdentifier()),
    "ETitan Precipitation Model": ETitanPrecipitationModel(MorphContourIdentifier(n_thresh=3, center_filter=10)),
    "Titan Precipitation Model": TitanPrecipitationModel(SimpleContourIdentifier()),
    "Ours Precipitation Model": OursPrecipitationModel(SimpleContourIdentifier()),
    "Adaptive Tracking Precipitation Model": AdaptiveTrackingPrecipitationModel(MorphContourIdentifier(n_thresh=3, center_filter=10))
}
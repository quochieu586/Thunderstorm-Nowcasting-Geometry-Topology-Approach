from dataclasses import dataclass
from typing import Annotated
from src.models import BasePrecipitationModel
from src.identification import BaseStormIdentifier, MorphContourIdentifier, SimpleContourIdentifier, HypothesisIdentifier, ClusterIdentifier
from src.cores.movement_estimate import TREC

@dataclass
class ModelSetting:
    identification_method: Annotated[str, "Method for identifying objects"]
    model_name: Annotated[str, "Name of the model"]
    model: BasePrecipitationModel
    max_velocity: int

@dataclass
class SensitivityAnalysisResults:
    pod_3: float
    far_3: float
    csi_3: float
    pod_5: float
    far_5: float
    csi_5: float
    object_consistency: float
    mean_duration: float
    linear_rmse: float
    optimal_tracking: float

    def print_result(self):
        print(f"Leading 4 time frames evaluation: POD@3: {self.pod_3:.4f}, FAR@3: {self.far_3:.4f}, CSI@3: {self.csi_3:.4f}")
        print(f"Leading 6 time frames evaluation: POD@5: {self.pod_5:.4f}, FAR@5: {self.far_5:.4f}, CSI@5: {self.csi_5:.4f}")
        print(f"Object consistency score: {self.object_consistency:.4f}")
        print(f"Mean duration of the tracks: {self.mean_duration:.4f} (minutes)")
        print(f"Linear RMSE: {self.linear_rmse:.4f} (km)")
        print(f"Optimal tracking evaluation score: {self.optimal_tracking:.4f}")

def create_identifier(identification_method: str, **args):
    if identification_method == "simple":
        return SimpleContourIdentifier()
    elif identification_method == "morphology":
        n_thresh = args.get("n_thresh", 5)
        center_filter = args.get("center_filter", 10)
        kernel_size = args.get("kernel_size", 5)
        return MorphContourIdentifier(
            n_thresh=n_thresh,
            center_filter=center_filter,
            kernel_size=kernel_size
        )
    elif identification_method == "hypothesis":
        distance_dbz_threshold = args.get("distance_dbz_threshold", 5)
        filter_center = args.get("filter_center", 10)
        return HypothesisIdentifier(distance_dbz_threshold=distance_dbz_threshold, filter_center=filter_center)
    elif identification_method == "cluster":
        filter_center = args.get("filter_center", 10)
        return ClusterIdentifier(filter_center=filter_center)
    else:
        raise ValueError(f"Unknown identification method: {identification_method}, expecting one of ['simple', 'morphology', 'hypothesis', 'cluster']")

def create_model(model_name: str, identifier: BaseStormIdentifier, **args) -> BasePrecipitationModel:
    max_velocity = args.get("max_velocity", 100)
    trec = None

    if model_name in ["etitan", "iscit", "stitan"]:
        trec = args.get("trec", TREC())

    if model_name == "ata":
        from src.models import AdaptiveTrackingPrecipitationModel
        return AdaptiveTrackingPrecipitationModel(identifier=identifier, max_velocity=max_velocity)
    elif model_name == "titan":
        from src.models import TitanPrecipitationModel
        return TitanPrecipitationModel(identifier=identifier, max_velocity=max_velocity)
    elif model_name == "etitan":
        from src.models import ETitanPrecipitationModel
        return ETitanPrecipitationModel(identifier=identifier, trec=trec, max_velocity=max_velocity)
    elif model_name == "iscit":
        from src.models import ISCITPrecipitationModel
        return ISCITPrecipitationModel(identifier=identifier, trec=trec, max_velocity=max_velocity)
    elif model_name == "stitan":
        from src.models import STitanPrecipitationModel
        return STitanPrecipitationModel(identifier=identifier, trec=trec, max_velocity=max_velocity)
    elif model_name == "ours":
        from src.models import OursPrecipitationModel
        return OursPrecipitationModel(identifier=identifier, max_velocity=max_velocity, velocity_estimate_weights=(0.5, 0.5))
    else:
        raise ValueError(f"Unknown model name: {model_name}, expecting one of ['ata', 'titan', 'etitan', 'iscit', 'stitan', 'ours']")
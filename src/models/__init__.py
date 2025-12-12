# from .simple import SimplePrecipitationModel
# from .ata_2021 import AdaptiveTrackingPrecipitationModel
# from .etitan_2009 import ETitanPrecipitationModel
# from .titan_1993 import TitanPrecipitationModel
from .ours import OursPrecipitationModel

from .base.model import BasePrecipitationModel

__all__ = ["BasePrecipitationModel", "OursPrecipitationModel"]
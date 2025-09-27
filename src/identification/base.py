from abc import ABC, abstractmethod
import numpy as np

from src.cores.contours import StormObject

class BaseStormIdentifier(ABC):
    @abstractmethod
    def identify_storm(self, image: np.ndarray, **args) -> list[StormObject]:
        """
            Identify storm objects from the image.

            Args:
                image (np.ndarray): The input image from which to identify storm objects.
                **args: Additional arguments that may be required for identification.
            Returns:
                List[StormObject]: A list of identified storm objects.
        """
        pass
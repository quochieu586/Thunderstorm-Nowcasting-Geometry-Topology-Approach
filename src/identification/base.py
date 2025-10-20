from abc import ABC, abstractmethod
import numpy as np

class BaseStormIdentifier(ABC):
    threshold: float
    filter_area: float

    def set_params(self, threshold: float, filter_area: float):
        self.threshold = threshold
        self.filter_area = filter_area

    @abstractmethod
    def identify_storm(self, image: np.ndarray, **args) -> list[np.ndarray]:
        """
            Identify storm objects from the image.

            Args:
                image (np.ndarray): The input image from which to identify storm objects.
                **args: Additional arguments that may be required for identification.
            Returns:
                List[np.ndarray]: A list of identified storm objects.
        """
        pass
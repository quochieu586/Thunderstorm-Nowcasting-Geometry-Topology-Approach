from abc import ABC, abstractmethod

from src.contours import StormObject

class BaseStormMotion(ABC):
    """
        Modeling the motion of storm objects between consecutive images.
    """
    @abstractmethod
    def estimate_motion(self, storm_object: StormObject) -> StormObject:
        """
            Estimate the motion vector (dx, dy) of storm objects between consecutive images.

            Args:
                images (list): A list of images (e.g., radar reflectivity maps) representing different time frames.
            Returns:
                tuple: A tuple representing the motion vector (dx, dy).
        """
        pass

    @abstractmethod
    def __add__(self, other: 'BaseStormMotion') -> 'BaseStormMotion':
        """
            Adding two motion together to get the combined motion.
        """
        pass

    @abstractmethod
    def update_motion(self, **args) -> 'BaseStormMotion':
        """
            Update the motion parameters.
        """
        pass
from abc import ABC, abstractmethod
from src.cores.base import StormsMap

class BasePrecipitationModeling(ABC):
    @abstractmethod
    def matching(self, previous_map: StormsMap, current_map: StormsMap) -> list[tuple[str, str]]:
        """
        Matching storms from previous map to current map. Return the list of matched storm IDs.
        """
        pass

    @abstractmethod
    def fit(self, train_data: list[StormsMap]):
        """
        Fit the model to the training data.
        """
        pass

    @abstractmethod
    def predict(self, previous_map: list[StormsMap]) -> StormsMap:
        """
        Predict the current storm map based on the previous storm maps.
        """
        pass
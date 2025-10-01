from abc import ABC, abstractmethod
from src.cores.contours import StormsMap

class BasePrecipitationModeling(ABC):
    @abstractmethod
    def fit(self, train_data: list[StormsMap]):
        pass

    @abstractmethod
    def predict(self, previous_map: list[StormsMap]) -> StormsMap:
        pass
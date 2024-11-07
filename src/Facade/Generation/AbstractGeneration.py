from abc import ABC, abstractmethod
import traci

class AbstractGeneration(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate(self):
        pass
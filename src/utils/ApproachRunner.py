from abc import abstractmethod, ABCMeta


class ApproachRunner(metaclass=ABCMeta):
    @abstractmethod
    def run(self, approach, snapshot, dump=False, **kwargs) -> dict:
        pass

    @abstractmethod
    def dump(self, approach, snapshot, predictions):
        pass

    @abstractmethod
    def load(self) -> object:
        pass

    @abstractmethod
    def evaluate(self, approach, snapshot) -> dict:
        pass

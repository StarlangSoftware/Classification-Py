from abc import abstractmethod

from Classification.Instance.Instance import Instance


class DistanceMetric(object):

    @abstractmethod
    def distance(self, instance1: Instance, instance2: Instance) -> float:
        pass

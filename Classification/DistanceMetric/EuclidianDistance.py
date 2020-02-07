from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.DistanceMetric.DistanceMetric import DistanceMetric
from Classification.Instance.Instance import Instance
import math


class EuclidianDistance(DistanceMetric):

    def distance(self, instance1: Instance, instance2: Instance) -> float:
        result = 0
        for i in range(instance1.attributeSize()):
            if isinstance(instance1.getAttribute(i), DiscreteAttribute) and \
                    isinstance(instance2.getAttribute(i), DiscreteAttribute):
                if instance1.getAttribute(i).getValue() is not None and \
                        instance1.getAttribute(i).getValue() != instance2.getAttribute(i).getValue():
                    result += 1
            else:
                if isinstance(instance1.getAttribute(i), ContinuousAttribute) and \
                        isinstance(instance2.getAttribute(i), ContinuousAttribute):
                    result += math.pow(instance1.getAttribute(i).getValue() - instance2.getAttribute(i).getValue(), 2)
        return result

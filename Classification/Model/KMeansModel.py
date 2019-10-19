from Math.DiscreteDistribution import DiscreteDistribution

from Classification.DistanceMetric.DistanceMetric import DistanceMetric
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.GaussianModel import GaussianModel


class KMeansModel(GaussianModel):

    """
    The constructor that sets the classMeans, priorDistribution and distanceMetric according to given inputs.

    PARAMETERS
    ----------
    priorDistribution : DiscreteDistribution
        DiscreteDistribution input.
    classMeans : InstanceList
        InstanceList of class means.
    distanceMetric : DistanceMetric
        DistanceMetric input.
    """
    def __init__(self, priorDistribution: DiscreteDistribution, classMeans: InstanceList, distanceMetric: DistanceMetric):
        self.classMeans = classMeans
        self.priorDistribution = priorDistribution
        self.distanceMetric = distanceMetric

    """
    The calculateMetric method takes an {@link Instance} and a String as inputs. It loops through the class means, if
    the corresponding class label is same as the given String it returns the negated distance between given instance and the
    current item of class means. Otherwise it returns the smallest negative number.

    PARAMETERS
    ----------
    instance : Instance
        Instance input.
    Ci : str      
        String input.
        
    RETURNS
    -------
    float
        The negated distance between given instance and the current item of class means.
    """
    def calculateMetric(self, instance: Instance, Ci: str) -> float:
        for i in range(self.classMeans.size()):
            if self.classMeans.get(i).getClassLabel() == Ci:
                return -self.distanceMetric.distance(instance, self.classMeans.get(i))
        return -1000000

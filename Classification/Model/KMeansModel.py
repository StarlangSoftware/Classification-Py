from Math.DiscreteDistribution import DiscreteDistribution

from Classification.DistanceMetric.DistanceMetric import DistanceMetric
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.GaussianModel import GaussianModel


class KMeansModel(GaussianModel):
    __class_means: InstanceList
    __distance_metric: DistanceMetric

    def __init__(self,
                 priorDistribution: DiscreteDistribution,
                 classMeans: InstanceList,
                 distanceMetric: DistanceMetric):
        """
        The constructor that sets the classMeans, priorDistribution and distanceMetric according to given inputs.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            DiscreteDistribution input.
        classMeans : InstanceList
            Class means.
        distanceMetric : DistanceMetric
            DistanceMetric input.
        """
        self.__class_means = classMeans
        self.prior_distribution = priorDistribution
        self.__distance_metric = distanceMetric

    def calculateMetric(self,
                        instance: Instance,
                        Ci: str) -> float:
        """
        The calculateMetric method takes an {@link Instance} and a String as inputs. It loops through the class means,
        if the corresponding class label is same as the given String it returns the negated distance between given
        instance and the current item of class means. Otherwise it returns the smallest negative number.

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
        for i in range(self.__class_means.size()):
            if self.__class_means.get(i).getClassLabel() == Ci:
                return -self.__distance_metric.distance(instance, self.__class_means.get(i))
        return -1000000

from Classification.DistanceMetric.DistanceMetric import DistanceMetric
from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Parameter.Parameter import Parameter


class KMeansParameter(Parameter):

    """
    Parameters of the Rocchio classifier.

    PARAMETERS
    ----------
    seed : int
        Seed is used for random number generation.
    distanceMetric : DistanceMetric
        distance metric used to calculate the distance between two instances.
    """
    def __init__(self, seed: int, distanceMetric = EuclidianDistance()):
        super().__init__(seed)
        self.distanceMetric = distanceMetric

    """
    Accessor for the distanceMetric.

    RETURNS
    -------
    DistanceMetric
        The distanceMetric.
    """
    def getDistanceMetric(self) -> DistanceMetric:
        return self.distanceMetric

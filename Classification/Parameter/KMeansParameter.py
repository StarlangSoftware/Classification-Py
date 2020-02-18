from Classification.DistanceMetric.DistanceMetric import DistanceMetric
from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Parameter.Parameter import Parameter


class KMeansParameter(Parameter):

    distanceMetric: DistanceMetric

    def __init__(self, seed: int, distanceMetric=EuclidianDistance()):
        """
        Parameters of the Rocchio classifier.

        PARAMETERS
        ----------
        seed : int
            Seed is used for random number generation.
        distanceMetric : DistanceMetric
            distance metric used to calculate the distance between two instances.
        """
        super().__init__(seed)
        self.distanceMetric = distanceMetric

    def getDistanceMetric(self) -> DistanceMetric:
        """
        Accessor for the distanceMetric.

        RETURNS
        -------
        DistanceMetric
            The distanceMetric.
        """
        return self.distanceMetric

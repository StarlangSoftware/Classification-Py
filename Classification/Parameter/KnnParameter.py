from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Parameter.KMeansParameter import KMeansParameter


class KnnParameter(KMeansParameter):

    __k: int

    """
    Parameters of the K-nearest neighbor classifier.

    PARAMETERS
    ----------
    seed : int
        Seed is used for random number generation.
    k : int
        Parameter of the K-nearest neighbor algorithm.
    distanceMetric : DistanceMetric
        Used to calculate the distance between two instances.
    """
    def __init__(self, seed: int, k: int, distanceMetric=EuclidianDistance()):
        super().__init__(seed, distanceMetric)
        self.__k = k

    """
    Accessor for the k.

    RETURNS
    -------
    int
        Value of the k.
    """
    def getK(self) -> int:
        return self.__k

from Classification.Parameter.KMeansParameter import KMeansParameter


class KnnParameter(KMeansParameter):

    """
    Parameters of the K-nearest neighbor classifier.

    PARAMETERS
    ----------
    seed : int
        Seed is used for random number generation.
    k : int
        Parameter of the K-nearest neighbor algorithm.
    """
    def __init__(self, seed: int, k: int):
        super().__init__(seed)
        self.k = k

    """
    Accessor for the k.

    RETURNS
    -------
    int
        Value of the k.
    """
    def getK(self) -> int:
        return self.k
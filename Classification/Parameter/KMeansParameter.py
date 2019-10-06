from Classification.Parameter.Parameter import Parameter


class KMeansParameter(Parameter):

    """
    Parameters of the Rocchio classifier.

    PARAMETERS
    ----------
    seed : int
        Seed is used for random number generation.
    """
    def __init__(self, seed: int):
        super().__init__(seed)


class Parameter(object):

    """
    Constructor of Parameter class which assigns given seed value to seed.

    PARAMETERS
    ----------
    seed : int
        Seed is used for random number generation.
    """
    def __init__(self, seed: int):
        self.seed = seed

    """
    Accessor for the seed.

    RETURNS
    -------
    int
        The seed.
    """
    def getSeed(self) -> int:
        return self.seed
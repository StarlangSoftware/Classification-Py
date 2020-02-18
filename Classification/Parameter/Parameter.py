class Parameter(object):

    seed: int

    def __init__(self, seed: int):
        """
        Constructor of Parameter class which assigns given seed value to seed.

        PARAMETERS
        ----------
        seed : int
            Seed is used for random number generation.
        """
        self.seed = seed

    def getSeed(self) -> int:
        """
        Accessor for the seed.

        RETURNS
        -------
        int
            The seed.
        """
        return self.seed

from Classification.Parameter.Parameter import Parameter


class BaggingParameter(Parameter):

    ensemble_size: int

    def __init__(self, seed: int, ensembleSize: int):
        """
        Parameters of the bagging trees algorithm.

        PARAMETERS
        ----------
        seed : int
            Seed is used for random number generation.
        ensembleSize : int
            The number of trees in the bagged forest.
        """
        super().__init__(seed)
        self.ensemble_size = ensembleSize

    def getEnsembleSize(self) -> int:
        """
        Accessor for the ensemble size.

        RETURNS
        -------
        int
            The ensemble size.
        """
        return self.ensemble_size

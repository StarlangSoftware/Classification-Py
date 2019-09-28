from Classification.Parameter.Parameter import Parameter


class BaggingParameter(Parameter):

    """
    Parameters of the bagging trees algorithm.

    PARAMETERS
    ----------
    seed : int
        Seed is used for random number generation.
    ensembleSize : int
        The number of trees in the bagged forest.
    """
    def __init__(self, seed: int, ensembleSize: int):
        super().__init__(seed)
        self.ensembleSize = ensembleSize

    """
    Accessor for the ensemble size.

    RETURNS
    -------
    int
        The ensemble size.
    """
    def getEnsembleSize(self) -> int:
        return self.ensembleSize
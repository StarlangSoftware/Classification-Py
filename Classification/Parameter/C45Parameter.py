from Classification.Parameter.Parameter import Parameter


class C45Parameter(Parameter):

    """
    Parameters of the C4.5 univariate decision tree classifier.

    PARAMETERS
    ----------
    seed : int
        Seed is used for random number generation.
    prune : bool
        Boolean value for prune.
    crossValidationRatio : float
        Double value for cross crossValidationRatio ratio.
    """
    def __init__(self, seed: int, prune: bool, crossValidationRatio: float):
        super().__init__(seed)
        self.prune = prune
        self.crossValidationRatio = crossValidationRatio

    """
    Accessor for the prune.

    RETURNS
    -------
    bool
        Prune.
    """
    def isPrune(self) -> bool:
        return self.prune

    """
    Accessor for the crossValidationRatio.

    RETURNS
    -------
    float
        crossValidationRatio.
    """
    def getCrossValidationRatio(self) -> float:
        return self.crossValidationRatio
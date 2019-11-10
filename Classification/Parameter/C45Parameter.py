from Classification.Parameter.Parameter import Parameter


class C45Parameter(Parameter):

    __prune: bool
    __crossValidationRatio: float

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
        self.__prune = prune
        self.__crossValidationRatio = crossValidationRatio

    """
    Accessor for the prune.

    RETURNS
    -------
    bool
        Prune.
    """
    def isPrune(self) -> bool:
        return self.__prune

    """
    Accessor for the crossValidationRatio.

    RETURNS
    -------
    float
        crossValidationRatio.
    """
    def getCrossValidationRatio(self) -> float:
        return self.__crossValidationRatio

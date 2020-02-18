from Classification.Parameter.Parameter import Parameter


class C45Parameter(Parameter):

    __prune: bool
    __crossValidationRatio: float

    def __init__(self, seed: int, prune: bool, crossValidationRatio: float):
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
        super().__init__(seed)
        self.__prune = prune
        self.__crossValidationRatio = crossValidationRatio

    def isPrune(self) -> bool:
        """
        Accessor for the prune.

        RETURNS
        -------
        bool
            Prune.
        """
        return self.__prune

    def getCrossValidationRatio(self) -> float:
        """
        Accessor for the crossValidationRatio.

        RETURNS
        -------
        float
            crossValidationRatio.
        """
        return self.__crossValidationRatio

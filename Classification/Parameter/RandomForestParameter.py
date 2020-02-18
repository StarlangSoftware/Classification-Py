from Classification.Parameter.BaggingParameter import BaggingParameter


class RandomForestParameter(BaggingParameter):

    __attributeSubsetSize: int

    def __init__(self, seed: int, ensembleSize: int, attributeSubsetSize: int):
        """
        Parameters of the random forest classifier.

        PARAMETERS
        ----------
        seed : int
            Seed is used for random number generation.
        ensembleSize : int
            The number of trees in the bagged forest.
        attributeSubsetSize : int
            Integer value for the size of attribute subset.
        """
        super().__init__(seed, ensembleSize)
        self.__attributeSubsetSize = attributeSubsetSize

    def getAttributeSubsetSize(self) -> int:
        """
        Accessor for the attributeSubsetSize.

        RETURNS
        -------
        int
            The attributeSubsetSize.
        """
        return self.__attributeSubsetSize

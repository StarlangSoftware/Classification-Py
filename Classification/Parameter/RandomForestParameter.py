from Classification.Parameter.BaggingParameter import BaggingParameter


class RandomForestParameter(BaggingParameter):

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
    def __init__(self, seed: int, ensembleSize: int, attributeSubsetSize: int):
        super().__init__(seed, ensembleSize)
        self.attributeSubsetSize = attributeSubsetSize

    """
    Accessor for the attributeSubsetSize.

    RETURNS
    -------
    int
        The attributeSubsetSize.
    """
    def getAttributeSubsetSize(self) -> int:
        return self.attributeSubsetSize
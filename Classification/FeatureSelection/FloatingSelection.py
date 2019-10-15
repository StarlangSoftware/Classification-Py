from Classification.FeatureSelection.FeatureSubSet import FeatureSubSet
from Classification.FeatureSelection.SubSetSelection import SubSetSelection


class FloatingSelection(SubSetSelection):

    """
    Constructor that creates a new {@link FeatureSubSet}.
    """
    def __init__(self):
        super().__init__(FeatureSubSet())

    """
    The operator method calls forward and backward methods.

    PARAMETERS
    ----------
    current : FeatureSubSet
        FeatureSubSet input.
    numberOfFeatures : int
        Indicates the indices of indexList.

    RETURNS
    -------
    list
        List of FeatureSubSet.
    """
    def operator(self, current: FeatureSubSet, numberOfFeatures: int) -> list:
        result = []
        self.forward(result, current, numberOfFeatures)
        self.backward(result, current)
        return result

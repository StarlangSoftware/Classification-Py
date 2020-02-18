from Classification.FeatureSelection.FeatureSubSet import FeatureSubSet
from Classification.FeatureSelection.SubSetSelection import SubSetSelection


class FloatingSelection(SubSetSelection):

    def __init__(self):
        """
        Constructor that creates a new {@link FeatureSubSet}.
        """
        super().__init__(FeatureSubSet())

    def operator(self, current: FeatureSubSet, numberOfFeatures: int) -> list:
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
        result = []
        self.forward(result, current, numberOfFeatures)
        self.backward(result, current)
        return result

from Classification.FeatureSelection.FeatureSubSet import FeatureSubSet
from Classification.FeatureSelection.SubSetSelection import SubSetSelection


class ForwardSelection(SubSetSelection):

    """
    Constructor that creates a new {@link FeatureSubSet}.
    """
    def __init__(self):
        super().__init__(FeatureSubSet())

    """
    The operator method calls forward method which starts with having no feature in the model. In each iteration,
    it keeps adding the features that are not currently listed.

    PARAMETERS
    ----------
    current : FeatureSubSet         
        FeatureSubset that will be added to new ArrayList.
    numberOfFeatures : int
        Indicates the indices of indexList.

    RETURNS
    -------
    list
        List of FeatureSubSets created from forward.
    """
    def operator(self, current: FeatureSubSet, numberOfFeatures: int) -> list:
        result = []
        self.forward(result, current, numberOfFeatures)
        return result

from Classification.FeatureSelection.FeatureSubSet import FeatureSubSet
from Classification.FeatureSelection.SubSetSelection import SubSetSelection


class BackwardSelection(SubSetSelection):

    def __init__(self, numberOfFeatures: int):
        """
        Constructor that creates a new FeatureSubSet and initializes indexList with given number of features.

        PARAMETERS
        ----------
        numberOfFeatures : int
            Indicates the indices of indexList.
        """
        super().__init__(FeatureSubSet(numberOfFeatures))

    def operator(self, current: FeatureSubSet, numberOfFeatures: int) -> list:
        """
        The operator method calls backward method which starts with all the features and removes the least significant
        feature at each iteration.

        PARAMETERS
        ----------
        current : FeatureSubSet
            FeatureSubset that will be added to new ArrayList.
        numberOfFeatures : int
            Indicates the indices of indexList.

        RETURNS
        -------
        list
            List of FeatureSubSets created from backward.
        """
        result = []
        self.backward(result, current)
        return result

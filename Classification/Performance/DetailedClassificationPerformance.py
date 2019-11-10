from Classification.Performance.ClassificationPerformance import ClassificationPerformance
from Classification.Performance.ConfusionMatrix import ConfusionMatrix


class DetailedClassificationPerformance(ClassificationPerformance):

    __confusionMatrix: ConfusionMatrix

    """
    A constructor that  sets the accuracy and errorRate as 1 - accuracy via given ConfusionMatrix and also sets the
    confusionMatrix.

    PARAMETERS
    ----------
    confusionMatrix : ConfusionMatrix
        ConfusionMatrix input.
    """
    def __init__(self, confusionMatrix: ConfusionMatrix):
        super().__init__(confusionMatrix.getAccuracy())
        self.__confusionMatrix = confusionMatrix

    """
    Accessor for the confusionMatrix.

    RETURNS
    -------
    ConfusionMatrix
        ConfusionMatrix.
    """
    def getConfusionMatrix(self) -> ConfusionMatrix:
        return self.__confusionMatrix

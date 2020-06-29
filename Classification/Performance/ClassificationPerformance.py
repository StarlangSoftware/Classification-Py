from Classification.Performance.Performance import Performance


class ClassificationPerformance(Performance):

    __accuracy: float

    def __init__(self, accuracy: float, errorRate: float = -1):
        """
        A constructor that sets the accuracy and errorRate via given input.

        PARAMETERS
        ----------
        accuracy : float
            Double value input.
        errorRate : float
            Double value input.
        """
        if errorRate == -1:
            self.errorRate = 1 - accuracy
        else:
            self.errorRate = errorRate
        self.__accuracy = accuracy

    def getAccuracy(self) -> float:
        """
        Accessor for the accuracy.

        RETURNS
        -------
        float
            Accuracy value.
        """
        return self.__accuracy

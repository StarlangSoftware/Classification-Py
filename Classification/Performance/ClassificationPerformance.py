from Classification.Performance.Performance import Performance


class ClassificationPerformance(Performance):

    """
    A constructor that sets the accuracy and errorRate via given input.

    PARAMETERS
    ----------
    accuracy : float 
        Double value input.
    errorRate : float
        Double value input.
    """
    def initWithBoth(self, accuracy: float, errorRate: float = -1):
        if errorRate == -1:
            self.errorRate = 1 - accuracy
        else:
            self.errorRate = errorRate
        self.accuracy = accuracy

    """
    Accessor for the accuracy.

    RETURNS
    -------
    float
        Accuracy value.
    """
    def getAccuracy(self) -> float:
        return self.accuracy
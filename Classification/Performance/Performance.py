class Performance(object):

    errorRate: float

    def __init__(self, errorRate: float):
        """
        Constructor that sets the error rate.

        PARAMETERS
        ----------
        errorRate : float
            Double input.
        """
        self.errorRate = errorRate

    def getErrorRate(self) -> float:
        """
        Accessor for the error rate.

        RETURNS
        -------
        float
            Double errorRate.
        """
        return self.errorRate

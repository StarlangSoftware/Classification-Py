class Performance(object):

    """
    Constructor that sets the error rate.

    PARAMETERS
    ----------
    errorRate : float
        Double input.
    """
    def __init__(self, errorRate: float):
        self.errorRate = errorRate

    """
    Accessor for the error rate.

    RETURNS
    -------
    float
        Double errorRate.
    """
    def getErrorRate(self) -> float:
        return self.errorRate
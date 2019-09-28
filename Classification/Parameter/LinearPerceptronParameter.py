from Classification.Parameter.Parameter import Parameter


class LinearPerceptronParameter(Parameter):

    """
    Parameters of the linear perceptron algorithm.

    PARAMETERS
    ----------
    seed : int
        Seed is used for random number generation.
    learningRate : float
        Double value for learning rate of the algorithm.
    etaDecrease : float
        Double value for decrease in eta of the algorithm.
    crossValidationRatio : float
        Double value for cross validation ratio of the algorithm.
    epoch : int
        Integer value for epoch number of the algorithm.
    """
    def __init__(self, seed: int, learningRate: float, etaDecrease: float, crossValidationRatio: float, epoch: int):
        super().__init__(seed)
        self.learningRate = learningRate
        self.etaDecrease = etaDecrease
        self.crossValidationRatio = crossValidationRatio
        self.epoch = epoch

    """
    Accessor for the learningRate.

    RETURNS
    -------
    float
        The learningRate.
    """
    def getLearningRate(self) -> float:
        return self.learningRate

    """
    Accessor for the etaDecrease.

    RETURNS
    -------
    float
        The etaDecrease.
    """
    def getEtaDecrease(self) -> float:
        return self.etaDecrease

    """
    Accessor for the crossValidationRatio.

    PARAMETERS
    ----------
    float
        The crossValidationRatio.
    """
    def getCrossValidationRatio(self) -> float:
        return self.crossValidationRatio

    """
    Accessor for the epoch.

    RETURNS
    -------
    int
        The epoch.
    """
    def getEpoch(self) -> int:
        return self.epoch
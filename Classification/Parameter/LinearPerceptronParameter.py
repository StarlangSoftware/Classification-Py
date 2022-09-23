from Classification.Parameter.Parameter import Parameter


class LinearPerceptronParameter(Parameter):

    learning_rate: float
    eta_decrease: float
    cross_validation_ratio: float
    __epoch: int

    def __init__(self,
                 seed: int,
                 learningRate: float,
                 etaDecrease: float,
                 crossValidationRatio: float,
                 epoch: int):
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
        super().__init__(seed)
        self.learning_rate = learningRate
        self.eta_decrease = etaDecrease
        self.cross_validation_ratio = crossValidationRatio
        self.__epoch = epoch

    def getLearningRate(self) -> float:
        """
        Accessor for the learningRate.

        RETURNS
        -------
        float
            The learningRate.
        """
        return self.learning_rate

    def getEtaDecrease(self) -> float:
        """
        Accessor for the etaDecrease.

        RETURNS
        -------
        float
            The etaDecrease.
        """
        return self.eta_decrease

    def getCrossValidationRatio(self) -> float:
        """
        Accessor for the crossValidationRatio.

        RETURNS
        ----------
        float
            The crossValidationRatio.
        """
        return self.cross_validation_ratio

    def getEpoch(self) -> int:
        """
        Accessor for the epoch.

        RETURNS
        -------
        int
            The epoch.
        """
        return self.__epoch

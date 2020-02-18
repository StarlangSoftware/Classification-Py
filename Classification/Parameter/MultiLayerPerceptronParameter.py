from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter


class MultiLayerPerceptronParameter(LinearPerceptronParameter):

    __hiddenNodes: int

    def __init__(self, seed: int, learningRate: float, etaDecrease: float, crossValidationRatio: float, epoch: int,
                 hiddenNodes: int):
        """
        Parameters of the multi layer perceptron algorithm.

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
        hiddenNodes : int
            Integer value for the number of hidden nodes.
        """
        super().__init__(seed, learningRate, etaDecrease, crossValidationRatio, epoch)
        self.__hiddenNodes = hiddenNodes

    def getHiddenNodes(self) -> int:
        """
        Accessor for the hiddenNodes.

        RETURNS
        -------
        int
            The hiddenNodes.
        """
        return self.__hiddenNodes

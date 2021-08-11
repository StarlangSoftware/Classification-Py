from Classification.Parameter.ActivationFunction import ActivationFunction
from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter


class MultiLayerPerceptronParameter(LinearPerceptronParameter):

    __hiddenNodes: int
    __activationFunction: ActivationFunction

    def __init__(self, seed: int, learningRate: float, etaDecrease: float, crossValidationRatio: float, epoch: int,
                 hiddenNodes: int, activationFunction: ActivationFunction):
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
        activationFunction : ActivationFunction
            Activation function.
        """
        super().__init__(seed, learningRate, etaDecrease, crossValidationRatio, epoch)
        self.__hiddenNodes = hiddenNodes
        self.__activationFunction = activationFunction

    def getHiddenNodes(self) -> int:
        """
        Accessor for the hiddenNodes.

        RETURNS
        -------
        int
            The hiddenNodes.
        """
        return self.__hiddenNodes

    def getActivationFunction(self) -> ActivationFunction:
        """
        Accessor for the activationFunction.

        RETURNS
        -------
        int
            The activation function.
        """
        return self.__activationFunction

from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.NeuralNetworkModel import NeuralNetworkModel
from Classification.Parameter.ActivationFunction import ActivationFunction
from Classification.Parameter.DeepNetworkParameter import DeepNetworkParameter
from Math.Matrix import Matrix
from Math.Vector import Vector
import copy

from Classification.Performance.ClassificationPerformance import ClassificationPerformance


class DeepNetworkModel(NeuralNetworkModel):

    __weights: list
    __hiddenLayerSize: int
    __activationFunction: ActivationFunction

    def __init__(self, trainSet: InstanceList, validationSet: InstanceList, parameters: DeepNetworkParameter):
        """
        Constructor that takes two InstanceList train set and validation set and DeepNetworkParameter as
        inputs. First it sets the class labels, their sizes as K and the size of the continuous attributes as d of given
        train set and allocates weights and sets the best weights. At each epoch, it shuffles the train set and loops
        through the each item of that train set, it multiplies the weights Matrix with input Vector than applies the
        sigmoid function and stores the result as hidden and add bias. Then updates weights and at the end it compares
        the performance of these weights with validation set. It updates the bestClassificationPerformance and
        bestWeights according to the current situation. At the end it updates the learning rate via etaDecrease value
        and finishes with clearing the weights.

        PARAMETERS
        ----------
        trainSet : InstanceList
            InstanceList to be used as trainSet.
        validationSet : InstanceList
            InstanceList to be used as validationSet.
        parameters : DeepNetworkParameter
            DeepNetworkParameter input.
        """
        super().__init__(trainSet)
        deltaWeights = []
        hidden = []
        hiddenBiased = []
        self.__activationFunction = parameters.getActivationFunction()
        self.__allocateWeights(parameters)
        bestWeights = self.__setBestWeights()
        bestClassificationPerformance = ClassificationPerformance(0.0)
        epoch = parameters.getEpoch()
        learningRate = parameters.getLearningRate()
        for i in range(epoch):
            trainSet.shuffle(parameters.getSeed())
            for j in range(trainSet.size()):
                self.createInputVector(trainSet.get(j))
                hidden.clear()
                hiddenBiased.clear()
                deltaWeights.clear()
                for k in range(self.__hiddenLayerSize):
                    if k == 0:
                        hidden.append(self.calculateHidden(self.x, self.__weights[k], self.__activationFunction))
                    else:
                        hidden.append(self.calculateHidden(hiddenBiased[k - 1], self.__weights[k], self.__activationFunction))
                    hiddenBiased.append(hidden[k].biased())
                rMinusY = self.calculateRMinusY(trainSet.get(j), hiddenBiased[self.__hiddenLayerSize - 1],
                                                self.__weights[len(self.__weights) - 1])
                deltaWeights.insert(0, Matrix(rMinusY, hiddenBiased[self.__hiddenLayerSize - 1]))
                for k in range(len(self.__weights) - 2, -1, -1):
                    if k == len(self.__weights) - 2:
                        tmph = self.__weights[k + 1].multiplyWithVectorFromLeft(rMinusY)
                    else:
                        tmph = self.__weights[k + 1].multiplyWithVectorFromLeft(tmpHidden)
                    tmph.remove(0)
                    if self.__activationFunction == ActivationFunction.SIGMOID:
                        oneMinusHidden = self.calculateOneMinusHidden(hidden[k])
                        activationDerivative = oneMinusHidden.elementProduct(hidden[k])
                    elif self.__activationFunction == ActivationFunction.TANH:
                        one = Vector(hidden[k].size(), 1.0)
                        hidden[k].tanh()
                        activationDerivative = one.difference(hidden[k].elementProduct(hidden[k]))
                    elif self.__activationFunction == ActivationFunction.RELU:
                        hidden[k].reluDerivative()
                        activationDerivative = hidden
                    tmpHidden = tmph.elementProduct(activationDerivative)
                    if k == 0:
                        deltaWeights.insert(0, Matrix(tmpHidden, self.x))
                    else:
                        deltaWeights.insert(0, Matrix(tmpHidden, hiddenBiased[k - 1]))
                for k in range(len(self.__weights)):
                    deltaWeights[k].multiplyWithConstant(learningRate)
                    self.__weights[k].add(deltaWeights[k])
            currentClassificationPerformance = self.testClassifier(validationSet)
            if currentClassificationPerformance.getAccuracy() > bestClassificationPerformance.getAccuracy():
                bestClassificationPerformance = currentClassificationPerformance
                bestWeights = self.__setBestWeights()
            learningRate *= parameters.getEtaDecrease()
        self.__weights.clear()
        for m in bestWeights:
            self.__weights.append(m)

    def __allocateWeights(self, parameters: DeepNetworkParameter):
        """
        The allocateWeights method takes DeepNetworkParameters as an input. First it adds random weights to the list
        of Matrix} weights' first layer. Then it loops through the layers and adds random weights till the last layer.
        At the end it adds random weights to the last layer and also sets the hiddenLayerSize value.

        PARAMETERS
        ----------
        parameters : DeepNetworkParameter
            DeepNetworkParameter input.
        """
        self.__weights = []
        self.__weights.append(self.allocateLayerWeights(parameters.getHiddenNodes(0), self.d + 1, parameters.getSeed()))
        for i in range(parameters.layerSize() - 1):
            self.__weights.append(self.allocateLayerWeights(parameters.getHiddenNodes(i + 1),
                                                            parameters.getHiddenNodes(i) + 1, parameters.getSeed()))
        self.__weights.append(self.allocateLayerWeights(self.K,
                                                        parameters.getHiddenNodes(parameters.layerSize() - 1) + 1, parameters.getSeed()))
        self.__hiddenLayerSize = parameters.layerSize()

    def __setBestWeights(self) -> list:
        """
        The setBestWeights method creates a list of Matrix as bestWeights and clones the values of weights list
        into this newly created list.

        RETURNS
        -------
        list
        A list clones from the weights ArrayList.
        """
        bestWeights = []
        for m in self.__weights:
            bestWeights.append(copy.deepcopy(m))
        return bestWeights

    def calculateOutput(self):
        """
        The calculateOutput method loops size of the weights times and calculate one hidden layer at a time and adds
        bias term. At the end it updates the output y value.
        """
        hiddenBiased = None
        for i in range(len(self.__weights) - 1):
            if i == 0:
                hidden = self.calculateHidden(self.x, self.__weights[i], self.__activationFunction)
            else:
                hidden = self.calculateHidden(hiddenBiased, self.__weights[i], self.__activationFunction)
            hiddenBiased = hidden.biased()
        self.y = self.__weights[len(self.__weights) - 1].multiplyWithVectorFromRight(hiddenBiased)

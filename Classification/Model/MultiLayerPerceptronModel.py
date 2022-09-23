from Math.Matrix import Matrix
from Math.Vector import Vector

from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.LinearPerceptronModel import LinearPerceptronModel
from Classification.Parameter.ActivationFunction import ActivationFunction
from Classification.Parameter.MultiLayerPerceptronParameter import MultiLayerPerceptronParameter
import copy

from Classification.Performance.ClassificationPerformance import ClassificationPerformance


class MultiLayerPerceptronModel(LinearPerceptronModel):

    __V: Matrix
    __activation_function: ActivationFunction

    def __allocateWeights(self,
                          H: int,
                          seed: int):
        """
        The allocateWeights method allocates layers' weights of Matrix W and V.

        PARAMETERS
        ----------
        H : int
            Integer value for weights.
        """
        self.W = self.allocateLayerWeights(row=H,
                                           column=self.d + 1,
                                           seed=seed)
        self.__V = self.allocateLayerWeights(row=self.K,
                                             column=H + 1,
                                             seed=seed)

    def __init__(self,
                 trainSet: InstanceList,
                 validationSet: InstanceList,
                 parameters: MultiLayerPerceptronParameter):
        """
        A constructor that takes InstanceLists as trainsSet and validationSet. It  sets the NeuralNetworkModel nodes
        with given InstanceList then creates an input vector by using given trainSet and finds error. Via the
        validationSet it finds the classification performance and reassigns the allocated weight Matrix with the matrix
        that has the best accuracy and the Matrix V with the best Vector input.

        PARAMETERS
        ----------
        trainSet : InstanceList
            InstanceList that is used to train.
        validationSet : InstanceList
            InstanceList that is used to validate.
        parameters : MultiLayerPerceptronParameter
            Multi layer perceptron parameters; seed, learningRate, etaDecrease, crossValidationRatio, epoch,
            hiddenNodes.
        """
        super().initWithTrainSet(trainSet)
        self.__activation_function = parameters.getActivationFunction()
        self.__allocateWeights(parameters.getHiddenNodes(), parameters.getSeed())
        best_w = copy.deepcopy(self.W)
        best_v = copy.deepcopy(self.__V)
        best_classification_performance = ClassificationPerformance(0.0)
        epoch = parameters.getEpoch()
        learning_rate = parameters.getLearningRate()
        for i in range(epoch):
            trainSet.shuffle(parameters.getSeed())
            for j in range(trainSet.size()):
                self.createInputVector(trainSet.get(j))
                hidden = self.calculateHidden(self.x, self.W, self.__activation_function)
                hidden_biased = hidden.biased()
                r_minus_y = self.calculateRMinusY(trainSet.get(j), hidden_biased, self.__V)
                delta_v = Matrix(r_minus_y, hidden_biased)
                tmp_h = self.__V.multiplyWithVectorFromLeft(r_minus_y)
                tmp_h.remove(0)
                if self.__activation_function == ActivationFunction.SIGMOID:
                    one_minus_hidden = self.calculateOneMinusHidden(hidden)
                    activation_derivative = one_minus_hidden.elementProduct(hidden)
                elif self.__activation_function == ActivationFunction.TANH:
                    one = Vector(hidden.size(), 1.0)
                    hidden.tanh()
                    activation_derivative = one.difference(hidden.elementProduct(hidden))
                elif self.__activation_function == ActivationFunction.RELU:
                    hidden.reluDerivative()
                    activation_derivative = hidden
                tmp_hidden = tmp_h.elementProduct(activation_derivative)
                delta_w = Matrix(tmp_hidden, self.x)
                delta_v.multiplyWithConstant(learning_rate)
                self.__V.add(delta_v)
                delta_w.multiplyWithConstant(learning_rate)
                self.W.add(delta_w)
            current_classification_performance = self.testClassifier(validationSet)
            if current_classification_performance.getAccuracy() > best_classification_performance.getAccuracy():
                best_classification_performance = current_classification_performance
                best_w = copy.deepcopy(self.W)
                best_v = copy.deepcopy(self.__V)
            learning_rate *= parameters.getEtaDecrease()
        self.W = best_w
        self.__V = best_v

    def calculateOutput(self):
        """
        The calculateOutput method calculates the forward single hidden layer by using Matrices W and V.
        """
        self.calculateForwardSingleHiddenLayer(W=self.W,
                                               V=self.__V,
                                               activationFunction=self.__activation_function)

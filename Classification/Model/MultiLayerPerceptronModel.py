from Math.Matrix import Matrix
from Math.Vector import Vector

from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
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

    def constructor2(self,
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
        super().__init__(trainSet)
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

    def constructor3(self, fileName: str):
        """
        Loads a multi-layer perceptron model from an input model file.
        :param fileName: Model file name.
        """
        inputFile = open(fileName, mode='r', encoding='utf-8')
        self.loadClassLabels(inputFile)
        self.W = self.loadMatrix(inputFile)
        self.__V = self.loadMatrix(inputFile)
        self.__activation_function = self.loadActivationFunction(inputFile)
        inputFile.close()

    def __init__(self,
                 trainSet: object = None,
                 validationSet: InstanceList = None,
                 parameters: MultiLayerPerceptronParameter = None):
        if isinstance(trainSet, InstanceList):
            self.constructor2(trainSet, validationSet, parameters)
        elif isinstance(trainSet, str):
            super().__init__()
            self.constructor3(trainSet)

    def calculateOutput(self):
        """
        The calculateOutput method calculates the forward single hidden layer by using Matrices W and V.
        """
        self.calculateForwardSingleHiddenLayer(W=self.W, V=self.__V, activationFunction=self.__activation_function)

    def train(self,
              trainSet: InstanceList,
              parameters: MultiLayerPerceptronParameter):
        """
        Training algorithm for the multilayer perceptron algorithm. 20 percent of the data is separated as
        cross-validation data used for selecting the best weights. 80 percent of the data is used for training the
        multilayer perceptron with gradient descent.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm
        parameters : MultiLayerPerceptronParameter
            Parameters of the multilayer perceptron.
        """
        partition = Partition(instanceList=trainSet, ratio=parameters.getCrossValidationRatio(),
                              seed=parameters.getSeed(), stratified=True)
        self.constructor2(trainSet=partition.get(1), validationSet=partition.get(0), parameters=parameters)

    def loadModel(self, fileName: str):
        """
        Loads the multi-layer perceptron model from an input file.
        :param fileName: File name of the multi-layer perceptron model.
        """
        self.constructor3(fileName)

from abc import abstractmethod
from io import TextIOWrapper

from Math.Matrix import Matrix
from Math.Vector import Vector

from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.ValidatedModel import ValidatedModel

import math

from Classification.Parameter.ActivationFunction import ActivationFunction


class NeuralNetworkModel(ValidatedModel):
    class_labels: list
    K: int
    d: int
    x: Vector
    y: Vector
    r: Vector

    @abstractmethod
    def calculateOutput(self):
        pass

    def __init__(self, trainSet: InstanceList = None):
        """
        Constructor that sets the class labels, their sizes as K and the size of the continuous attributes as d.

        PARAMETERS
        ----------
        trainSet : InstanceList
            InstanceList to use as train set.
        """
        if trainSet is not None:
            self.class_labels = trainSet.getDistinctClassLabels()
            self.K = len(self.class_labels)
            self.d = trainSet.get(0).continuousAttributeSize()

    def allocateLayerWeights(self,
                             row: int,
                             column: int,
                             seed: int) -> Matrix:
        """
        The allocateLayerWeights method returns a new Matrix with random weights.

        PARAMETERS
        ----------
        row : int
            Number of rows.
        column : int
            Number of columns.
        seed : int
            Seed for initialization of random function.

        RETURNS
        -------
        Matrix
            Matrix with random weights.
        """
        matrix = Matrix(row=row,
                        col=column,
                        minValue=-0.01,
                        maxValue=+0.01,
                        seed=seed)
        return matrix

    def normalizeOutput(self, o: Vector) -> Vector:
        """
        The normalizeOutput method takes an input {@link Vector} o, gets the result for e^o of each element of o,
        then sums them up. At the end, divides the each e^o by the summation.

        PARAMETERS
        ----------
        o : Vector
            Vector to normalize.

        RETURNS
        -------
        Vector
            Normalized vector.
        """
        total = 0.0
        values = []
        for i in range(o.size()):
            if o.getValue(i) > 500:
                total += math.exp(500)
            else:
                total += math.exp(o.getValue(i))
        for i in range(o.size()):
            if o.getValue(i) > 500:
                values.append(math.exp(500) / total)
            else:
                values.append(math.exp(o.getValue(i)) / total)
        return Vector(values)

    def createInputVector(self, instance: Instance):
        """
        The createInputVector method takes an Instance as an input. It converts given Instance to the Vector
        and insert 1.0 to the first element.

        PARAMETERS
        ----------
        instance : Instance
            Instance to insert 1.0.
        """
        self.x = instance.toVector()
        self.x.insert(0, 1.0)

    def calculateHidden(self, input: Vector, weights: Matrix, activationFunction: ActivationFunction) -> Vector:
        """
        The calculateHidden method takes a {@link Vector} input and {@link Matrix} weights, It multiplies the weights
        Matrix with given input Vector than applies the sigmoid function and returns the result.

        PARAMETERS
        ----------
        input : Vector
            Vector to multiply weights.
        weights : Matrix
            Matrix is multiplied with input Vector.
        activationFunction : ActivationFunction
            Activation function

        RETURNS
        -------
        Vector
            Result of sigmoid function.
        """
        z = weights.multiplyWithVectorFromRight(input)
        if activationFunction == ActivationFunction.SIGMOID:
            z.sigmoid()
        elif activationFunction == ActivationFunction.TANH:
            z.tanh()
        elif activationFunction == ActivationFunction.RELU:
            z.relu()
        return z

    def calculateOneMinusHidden(self, hidden: Vector) -> Vector:
        """
        The calculateOneMinusHidden method takes a {@link java.util.Vector} as input. It creates a Vector of ones and
         returns the difference between given Vector.

        PARAMETERS
        ----------
        hidden : Vector
            Vector to find difference.

        RETURNS
        -------
        Vector
            Returns the difference between ones Vector and input Vector.
        """
        one = Vector()
        one.initAllSame(hidden.size(), 1.0)
        return one.difference(hidden)

    def calculateForwardSingleHiddenLayer(self, W: Matrix, V: Matrix, activationFunction: ActivationFunction):
        """
        The calculateForwardSingleHiddenLayer method takes two matrices W and V. First it multiplies W with x, then
        multiplies V with the result of the previous multiplication.

        PARAMETERS
        ----------
        W : Matrix
            Matrix to multiply with x.
        V : Matrix
            Matrix to multiply.
        activationFunction : ActivationFunction
            Activation function
        """
        hidden = self.calculateHidden(self.x, W, activationFunction)
        hidden_biased = hidden.biased()
        self.y = V.multiplyWithVectorFromRight(hidden_biased)

    def calculateRMinusY(self, instance: Instance, inputVector: Vector, weights: Matrix) -> Vector:
        """
        The calculateRMinusY method creates a new Vector with given Instance, then it multiplies given
        input Vector with given weights Matrix. After normalizing the output, it return the difference between the newly
        created Vector and normalized output.

        PARAMETERS
        ----------
        instance : Instance
            Instance is used to get class labels.
        inputVector : Vector
            Vector to multiply weights.
        weights : Matrix
            Matrix of weights

        RETURNS
        -------
        Vector
            Difference between newly created Vector and normalized output.
        """
        r = Vector()
        r.initAllZerosExceptOne(self.K, self.class_labels.index(instance.getClassLabel()), 1.0)
        o = weights.multiplyWithVectorFromRight(inputVector)
        y = self.normalizeOutput(o)
        return r.difference(y)

    def predictWithCompositeInstance(self, possibleClassLabels: list) -> str:
        """
        The predictWithCompositeInstance method takes an ArrayList possibleClassLabels. It returns the class label
        which has the maximum value of y.

        PARAMETERS
        ----------
        possibleClassLabels : list
            List that has the class labels.

        RETURNS
        -------
        str
            The class label which has the maximum value of y.
        """
        predicted_class = possibleClassLabels[0]
        maxY = -100000000
        for i in range(len(self.class_labels)):
            if self.class_labels[i] in possibleClassLabels and self.y.getValue(i) > maxY:
                maxY = self.y.getValue(i)
                predicted_class = self.class_labels[i]
        return predicted_class

    def predict(self, instance: Instance) -> str:
        """
        The predict method takes an Instance as an input, converts it to a Vector and calculates the Matrix y by
        multiplying Matrix W with Vector x. Then it returns the class label which has the maximum y value.

        PARAMETERS
        ----------
        instance : Instance
            Instance to predict.

        RETURNS
        -------
        str
            The class label which has the maximum y.
        """
        self.createInputVector(instance)
        self.calculateOutput()
        if isinstance(instance, CompositeInstance):
            return self.predictWithCompositeInstance(instance.getPossibleClassLabels())
        else:
            return self.class_labels[self.y.maxIndex()]

    def predictProbability(self, instance: Instance) -> dict:
        """
        Calculates the posterior probability distribution for the given instance according to neural network model.
        :param instance: Instance for which posterior probability distribution is calculated.
        :return: Posterior probability distribution for the given instance.
        """
        self.createInputVector(instance)
        self.calculateOutput()
        result = {}
        for i in range(len(self.class_labels)):
            result[self.class_labels[i]] = self.y.getValue(i)
        return result

    def loadClassLabels(self, inputFile: TextIOWrapper):
        """
        Loads the class labels from input model file.
        :param inputFile: Input model file.
        """
        items = inputFile.readline().strip().split(" ")
        self.K = int(items[0])
        self.d = int(items[1])
        self.class_labels = list()
        for i in range(self.K):
            self.class_labels.append(inputFile.readline().strip())

    def loadActivationFunction(self, inputFile: TextIOWrapper):
        """
        Loads the activation function from an input model file.
        :param inputFile: Input model file.
        :return: Activation function read.
        """
        line = inputFile.readline().strip()
        if line == "TANH":
            return ActivationFunction.TANH
        elif line == "RELU":
            return ActivationFunction.RELU
        else:
            return ActivationFunction.SIGMOID

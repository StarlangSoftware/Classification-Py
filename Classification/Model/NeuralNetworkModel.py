from abc import abstractmethod

from Math.Matrix import Matrix
from Math.Vector import Vector

from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.ValidatedModel import ValidatedModel

import math


class NeuralNetworkModel(ValidatedModel):
    classLabels: list
    K: int
    d: int
    x: Vector
    y: Vector
    r: Vector

    @abstractmethod
    def calculateOutput(self):
        pass

    """
    Constructor that sets the class labels, their sizes as K and the size of the continuous attributes as d.

    PARAMETERS
    ----------
    trainSet : InstanceList
        InstanceList to use as train set.
    """

    def __init__(self, trainSet: InstanceList):
        self.classLabels = trainSet.getDistinctClassLabels()
        self.K = len(self.classLabels)
        self.d = trainSet.get(0).continuousAttributeSize()

    """
    The allocateLayerWeights method returns a new Matrix with random weights.

    PARAMETERS
    ----------
    row : int
        Number of rows.
    column : int
        Number of columns.
        
    RETURNS
    -------
    Matrix
        Matrix with random weights.
    """

    def allocateLayerWeights(self, row: int, column: int) -> Matrix:
        matrix = Matrix(row, column, -0.01, +0.01)
        return matrix

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

    def normalizeOutput(self, o: Vector) -> Vector:
        total = 0.0
        values = []
        for i in range(len(values)):
            total += math.exp(o.getValue(i))
        for i in range(len(values)):
            values.append(math.exp(o.getValue(i)) / total)
        return Vector(values)

    """
    The createInputVector method takes an Instance as an input. It converts given Instance to the Vector
    and insert 1.0 to the first element.

    PARAMETERS
    ----------
    instance : Instance
        Instance to insert 1.0.
    """

    def createInputVector(self, instance: Instance):
        self.x = instance.toVector()
        self.x.insert(0, 1.0)

    """
    The calculateHidden method takes a {@link Vector} input and {@link Matrix} weights, It multiplies the weights
    Matrix with given input Vector than applies the sigmoid function and returns the result.

    PARAMETERS
    ----------
    input : Vector  
        Vector to multiply weights.
    weights : Matrix
        Matrix is multiplied with input Vector.
        
    RETURNS
    -------
    Vector
        Result of sigmoid function.
    """

    def calculateHidden(self, input: Vector, weights: Matrix) -> Vector:
        z = weights.multiplyWithVectorFromRight(input)
        z.sigmoid()
        return z

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

    def calculateOneMinusHidden(self, hidden: Vector) -> Vector:
        one = Vector()
        one.initAllSame(hidden.size(), 1.0)
        return one.difference(hidden)

    """
    The calculateForwardSingleHiddenLayer method takes two matrices W and V. First it multiplies W with x, then
    multiplies V with the result of the previous multiplication.

    PARAMETERS
    ----------
    W : Matrix
        Matrix to multiply with x.
    V : Matrix
        Matrix to multiply.
    """

    def calculateForwardSingleHiddenLayer(self, W: Matrix, V: Matrix):
        hidden = self.calculateHidden(self.x, W)
        hiddenBiased = hidden.biased()
        self.y = V.multiplyWithVectorFromRight(hiddenBiased)

    """
    The calculateRMinusY method creates a new Vector with given Instance, then it multiplies given
    input Vector with given weights Matrix. After normalizing the output, it return the difference between the newly 
    created Vector and normalized output.

    PARAMETERS
    ----------
    instance : Instance
        Instance is used to get class labels.
    input : Vector   
        Vector to multiply weights.
    weights : Matrix 
        Matrix of weights
        
    RETURNS
    -------
    Vector
        Difference between newly created Vector and normalized output.
    """

    def calculateRMinusY(self, instance: Instance, inputVector: Vector, weights: Matrix) -> Vector:
        r = Vector()
        r.initAllZerosExceptOne(self.K, self.classLabels.index(instance.getClassLabel()), 1.0)
        o = weights.multiplyWithVectorFromRight(inputVector)
        y = self.normalizeOutput(o)
        return r.difference(y)

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

    def predictWithCompositeInstance(self, possibleClassLabels: list) -> str:
        predictedClass = possibleClassLabels[0]
        maxY = -100000000
        for i in range(len(self.classLabels)):
            if self.classLabels[i] in possibleClassLabels and self.y.getValue(i) > maxY:
                maxY = self.y.getValue(i)
                predictedClass = self.classLabels[i]
        return predictedClass

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

    def predict(self, instance: Instance) -> str:
        self.createInputVector(instance)
        self.calculateOutput()
        if isinstance(instance, CompositeInstance):
            return self.predictWithCompositeInstance(instance.getPossibleClassLabels())
        else:
            return self.classLabels[self.y.maxIndex()]

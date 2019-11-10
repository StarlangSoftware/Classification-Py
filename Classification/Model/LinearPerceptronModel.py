from Math.Matrix import Matrix

from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.NeuralNetworkModel import NeuralNetworkModel
from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter
import copy

from Classification.Performance.ClassificationPerformance import ClassificationPerformance


class LinearPerceptronModel(NeuralNetworkModel):

    W: Matrix

    def initWithTrainSet(self, trainSet: InstanceList):
        super().__init__(trainSet)

    """
    Constructor that takes InstanceLists as trainsSet and validationSet. Initially it allocates layer weights,
    then creates an input vector by using given trainSet and finds error. Via the validationSet it finds the
    classification performance and at the end it reassigns the allocated weight Matrix with the matrix that has the best
    accuracy.

    PARAMETERS
    ----------
    trainSet : InstanceList
        InstanceList that is used to train.
    validationSet : InstanceList
        InstanceList that is used to validate.
    parameters : LinearPerceptronParameter
        Linear perceptron parameters; learningRate, etaDecrease, crossValidationRatio, epoch.
    """
    def __init__(self, trainSet: InstanceList, validationSet: InstanceList, parameters: LinearPerceptronParameter):
        super().__init__(trainSet)
        self.W = self.allocateLayerWeights(self.K, self.d + 1)
        bestW = copy.deepcopy(self.W)
        bestClassificationPerformance = ClassificationPerformance(0.0)
        epoch = parameters.getEpoch()
        learningRate = parameters.getLearningRate()
        for i in range(epoch):
            trainSet.shuffle(parameters.getSeed())
            for j in range(trainSet.size()):
                self.createInputVector(trainSet.get(j))
                rMinusY = self.calculateRMinusY(trainSet.get(j), self.x, self.W)
                deltaW = rMinusY.multiplyWithVector(self.x)
                deltaW.multiplyWithConstant(learningRate)
                self.W.add(deltaW)
            currentClassificationPerformance = self.testClassifier(validationSet)
            if currentClassificationPerformance.getAccuracy() > bestClassificationPerformance.getAccuracy():
                bestClassificationPerformance = currentClassificationPerformance
                bestW = copy.deepcopy(self.W)
            learningRate *= parameters.getEtaDecrease()
        self.W = bestW

    """
    The calculateOutput method calculates the Matrix y by multiplying Matrix W with Vector x.
    """
    def calculateOutput(self):
        self.y = self.W.multiplyWithVectorFromRight(self.x)

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

    def __init__(self,
                 trainSet: InstanceList,
                 validationSet: InstanceList,
                 parameters: LinearPerceptronParameter):
        """
        Constructor that takes InstanceLists as trainsSet and validationSet. Initially it allocates layer weights,
        then creates an input vector by using given trainSet and finds error. Via the validationSet it finds the
        classification performance and at the end it reassigns the allocated weight Matrix with the matrix that has the
        best accuracy.

        PARAMETERS
        ----------
        trainSet : InstanceList
            InstanceList that is used to train.
        validationSet : InstanceList
            InstanceList that is used to validate.
        parameters : LinearPerceptronParameter
            Linear perceptron parameters; learningRate, etaDecrease, crossValidationRatio, epoch.
        """
        super().__init__(trainSet)
        self.W = self.allocateLayerWeights(row=self.K,
                                           column=self.d + 1,
                                           seed=parameters.getSeed())
        best_w = copy.deepcopy(self.W)
        best_classification_performance = ClassificationPerformance(0.0)
        epoch = parameters.getEpoch()
        learning_rate = parameters.getLearningRate()
        for i in range(epoch):
            trainSet.shuffle(parameters.getSeed())
            for j in range(trainSet.size()):
                self.createInputVector(trainSet.get(j))
                r_minus_y = self.calculateRMinusY(trainSet.get(j), self.x, self.W)
                delta_w = Matrix(r_minus_y, self.x)
                delta_w.multiplyWithConstant(learning_rate)
                self.W.add(delta_w)
            current_classification_performance = self.testClassifier(validationSet)
            if current_classification_performance.getAccuracy() > best_classification_performance.getAccuracy():
                best_classification_performance = current_classification_performance
                best_w = copy.deepcopy(self.W)
            learning_rate *= parameters.getEtaDecrease()
        self.W = best_w

    def calculateOutput(self):
        """
        The calculateOutput method calculates the Matrix y by multiplying Matrix W with Vector x.
        """
        self.y = self.W.multiplyWithVectorFromRight(self.x)

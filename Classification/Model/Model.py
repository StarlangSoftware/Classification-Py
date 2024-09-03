from abc import abstractmethod
from io import TextIOWrapper

from Math.DiscreteDistribution import DiscreteDistribution
from Math.Matrix import Matrix

from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.DiscreteIndexedAttribute import DiscreteIndexedAttribute
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Parameter.Parameter import Parameter
from Classification.Performance.ConfusionMatrix import ConfusionMatrix
from Classification.Performance.DetailedClassificationPerformance import DetailedClassificationPerformance
from Classification.Performance.Performance import Performance


class Model(object):

    @abstractmethod
    def train(self,
              trainSet: InstanceList,
              parameters: Parameter):
        pass

    @abstractmethod
    def loadModel(self, fileName: str):
        pass

    @abstractmethod
    def predict(self, instance: Instance) -> str:
        """
         An abstract predict method that takes an Instance as an input.

        PARAMETERS
        ----------
        instance : Instance
            Instance to make prediction.

        RETURNS
        -------
        str
            The class label as a String.
        """
        pass

    @abstractmethod
    def predictProbability(self, instance: Instance) -> dict:
        pass

    def loadInstance(self, line: str, attributeTypes: list) -> Instance:
        items = line.split(",")
        instance = Instance(items[len(items) - 1])
        for i in range(len(items) - 1):
            if attributeTypes[i] == "DISCRETE":
                instance.addDiscreteAttribute(items[i])
            elif attributeTypes[i] == "CONTINUOUS":
                instance.addContinuousAttribute(float(items[i]))
        return instance

    def loadMatrix(self, inputFile: TextIOWrapper) -> Matrix:
        items = inputFile.readline().strip().split(" ")
        matrix = Matrix(int(items[0]), int(items[1]))
        for j in range(matrix.getRow()):
            line = inputFile.readline().strip()
            items = line.split(" ")
            for k in range(matrix.getColumn()):
                matrix.setValue(j, k, float(items[k]))
        return matrix

    @staticmethod
    def loadClassDistribution(inputFile: TextIOWrapper) -> DiscreteDistribution:
        distribution = DiscreteDistribution()
        size = int(inputFile.readline().strip())
        for i in range(size):
            line = inputFile.readline().strip()
            items = line.split(" ")
            count = int(items[1])
            for j in range(count):
                distribution.addItem(items[0])
        return distribution

    def loadInstanceList(self, inputFile: TextIOWrapper) -> InstanceList:
        """
        Loads an instance list from an input model file.
        :param inputFile: Input model file.
        :return: Instance list read from an input model file.
        """
        types = inputFile.readline().strip().split(" ")
        instance_count = int(inputFile.readline().strip())
        instance_list = InstanceList()
        for i in range(instance_count):
            instance_list.add(self.loadInstance(inputFile.readline().strip(), types))
        return instance_list

    def discreteCheck(self, instance: Instance) -> bool:
        """
        Checks given instance's attribute and returns true if it is a discrete indexed attribute, false otherwise.

        PARAMETERS
        ----------
        instance Instance to check.

        RETURNS
        -------
        bool
            True if instance is a discrete indexed attribute, false otherwise.
        """
        for i in range(instance.attributeSize()):
            if isinstance(instance.getAttribute(i), DiscreteAttribute) and not isinstance(instance.getAttribute(i),
                                                                                          DiscreteIndexedAttribute):
                return False
        return True

    def test(self, testSet: InstanceList) -> Performance:
        """
        TestClassification an instance list with the current model.

        PARAMETERS
        ----------
        testSet : InstanceList
            Test data (list of instances) to be tested.

        RETURNS
        -------
        Performance
            The accuracy (and error) of the model as an instance of Performance class.
        """
        class_labels = testSet.getUnionOfPossibleClassLabels()
        confusion = ConfusionMatrix(class_labels)
        for i in range(testSet.size()):
            instance = testSet.get(i)
            confusion.classify(instance.getClassLabel(), self.predict(instance))
        return DetailedClassificationPerformance(confusion)

    def singleRun(self,
                  parameter: Parameter,
                  trainSet: InstanceList,
                  testSet: InstanceList) -> Performance:
        """
        Runs current classifier with the given train and test data.

        PARAMETERS
        ----------
        parameter : Parameter
            Parameter of the classifier to be trained.
        trainSet : InstanceList
            Training data to be used in training the classifier.
        testSet : InstanceList
            Test data to be tested after training the model.

        RETURNS
        -------
        Performance
            The accuracy (and error) of the trained model as an instance of Performance class.
        """
        self.train(trainSet, parameter)
        return self.test(testSet)

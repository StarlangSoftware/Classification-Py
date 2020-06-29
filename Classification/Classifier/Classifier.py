from abc import abstractmethod
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Parameter.Parameter import Parameter
from Classification.Instance.Instance import Instance
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.DiscreteIndexedAttribute import DiscreteIndexedAttribute
from Classification.Performance.Performance import Performance
from Classification.Performance.ConfusionMatrix import ConfusionMatrix
from Classification.Performance.DetailedClassificationPerformance import DetailedClassificationPerformance
from Classification.Model.Model import Model
from DataStructure.CounterHashMap import CounterHashMap


class Classifier(object):

    model: Model

    @abstractmethod
    def train(self, trainSet: InstanceList, parameters: Parameter):
        pass

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
        testSet : InstaceList
            Test data (list of instances) to be tested.

        RETURNS
        -------
        Performance
            The accuracy (and error) of the model as an instance of Performance class.
        """
        classLabels = testSet.getUnionOfPossibleClassLabels()
        confusion = ConfusionMatrix(classLabels)
        for i in range(testSet.size()):
            instance = testSet.get(i)
            confusion.classify(instance.getClassLabel(), self.model.predict(instance))
        return DetailedClassificationPerformance(confusion)

    def singleRun(self, parameter: Parameter, trainSet: InstanceList, testSet: InstanceList) -> Performance:
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

    def getModel(self) -> Model:
        """
        Accessor for the model.

        RETURNS
        -------
        Model
            Model.
        """
        return self.model

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

    @abstractmethod
    def train(self, trainSet: InstanceList, parameters: Parameter):
        pass

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
    def discreteCheck(self, instance: Instance) -> bool:
        for i in range(instance.attributeSize()):
            if isinstance(instance.getAttribute(i), DiscreteAttribute) and not isinstance(instance.getAttribute(i), DiscreteIndexedAttribute):
                return False
        return True

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
    def test(self, testSet: InstanceList) -> Performance:
        classLabels = testSet.getUnionOfPossibleClassLabels()
        confusion = ConfusionMatrix(classLabels)
        for i in range(testSet.size()):
            instance = testSet.get(i)
            confusion.classify(instance.getClassLabel(), self.model.predict(instance))
        return DetailedClassificationPerformance(confusion)

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
    def singleRun(self, parameter: Parameter, trainSet: InstanceList, testSet: InstanceList) -> Performance:
        self.train(trainSet, parameter)
        return self.test(testSet)

    """
    Accessor for the model.

    RETURNS
    -------
    Model
        Model.
    """
    def getModel(self) -> Model:
        return self.model

    """
    Given an array of class labels, returns the maximum occurred one.

    PARAMETERS
    ----------
    classLabels : list
        An array of class labels.
        
    RETURNS
    -------
    str
        The class label that occurs most in the array of class labels (mod of class label list).
    """
    def getMaximum(self, classLabels: list) -> str:
        frequencies = CounterHashMap()
        for label in classLabels:
            frequencies.put(label)
        return frequencies.max()
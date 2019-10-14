from Classification.Classifier.Classifier import Classifier
from Classification.Parameter.Parameter import Parameter
from Classification.DataSet.DataSet import DataSet


class Experiment(object):

    """
    Constructor for a specific machine learning experiment

    PARAMETERS
    ----------
    classifier : Classifier
        Classifier used in the machine learning experiment
    parameter : Parameter
        Parameter(s) of the classifier.
    dataSet : DataSet
        DataSet on which the classifier is run.
    """
    def __init__(self, classifier: Classifier, parameter: Parameter, dataSet: DataSet):
        self.classifier = classifier
        self.parameter = parameter
        self.dataSet = dataSet

    """
    Accessor for the classifier attribute.
    
    RETURNS
    -------
    Classifier
        Classifier attribute.
    """
    def getClassifier(self) -> Classifier:
        return self.classifier

    """
    Accessor for the parameter attribute.

    RETURNS
    -------
    Parameter
        Parameter attribute.
    """

    def getParameter(self) -> Parameter:
        return self.parameter

    """
    Accessor for the dataSet attribute.

    RETURNS
    -------
    DataSet
        DataSet attribute.
    """

    def getDataSet(self) -> DataSet:
        return self.dataSet

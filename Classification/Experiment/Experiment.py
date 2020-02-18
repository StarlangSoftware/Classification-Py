from __future__ import annotations
from Classification.Classifier.Classifier import Classifier
from Classification.FeatureSelection.FeatureSubSet import FeatureSubSet
from Classification.Parameter.Parameter import Parameter
from Classification.DataSet.DataSet import DataSet


class Experiment(object):

    __classifier: Classifier
    __parameter: Parameter
    __dataSet: DataSet

    def __init__(self, classifier: Classifier, parameter: Parameter, dataSet: DataSet):
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
        self.__classifier = classifier
        self.__parameter = parameter
        self.__dataSet = dataSet

    def getClassifier(self) -> Classifier:
        """
        Accessor for the classifier attribute.

        RETURNS
        -------
        Classifier
            Classifier attribute.
        """
        return self.__classifier

    def getParameter(self) -> Parameter:
        """
        Accessor for the parameter attribute.

        RETURNS
        -------
        Parameter
            Parameter attribute.
        """
        return self.__parameter

    def getDataSet(self) -> DataSet:
        """
        Accessor for the dataSet attribute.

        RETURNS
        -------
        DataSet
            DataSet attribute.
        """
        return self.__dataSet

    def featureSelectedExperiment(self, featureSubSet: FeatureSubSet) -> Experiment:
        """
        Construct and returns a feature selection experiment.

        PARAMETERS
        ----------
        featureSubSet : FeatureSubSet
            Feature subset used in the feature selection experiment

        RETURNS
        -------
        Experiment
            Experiment constructed
        """
        return Experiment(self.__classifier, self.__parameter, self.__dataSet.getSubSetOfFeatures(featureSubSet))

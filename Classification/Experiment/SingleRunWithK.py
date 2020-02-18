from Sampling.CrossValidation import CrossValidation
from Sampling.KFoldCrossValidation import KFoldCrossValidation

from Classification.Classifier.Classifier import Classifier
from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.SingleRun import SingleRun
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Parameter.Parameter import Parameter
from Classification.Performance.Performance import Performance


class SingleRunWithK(SingleRun):

    __K: int

    def __init__(self, K: int):
        """
        Constructor for SingleRunWithK class. Basically sets K parameter of the K-fold cross-validation.

        PARAMETERS
        ----------
        K : int
            K of the K-fold cross-validation.
        """
        self.__K = K

    def runExperiment(self, classifier: Classifier, parameter: Parameter, crossValidation: CrossValidation):
        trainSet = InstanceList(crossValidation.getTrainFold(0))
        testSet = InstanceList(crossValidation.getTestFold(0))
        return classifier.singleRun(parameter, trainSet, testSet)

    def execute(self, experiment: Experiment) -> Performance:
        """
        Execute Single K-fold cross-validation with the given classifier on the given data set using the given
        parameters.

        PARAMETERS
        -----
        experiment : Experiment
            Experiment to be run.

        RETURNS
        -------
        Performance
            A Performance instance.
        """
        crossValidation = KFoldCrossValidation(experiment.getDataSet().getInstances(), self.__K,
                                               experiment.getParameter().getSeed())
        return self.runExperiment(experiment.getClassifier(), experiment.getParameter(), crossValidation)

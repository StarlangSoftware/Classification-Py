from Sampling.CrossValidation import CrossValidation
from Sampling.KFoldCrossValidation import KFoldCrossValidation

from Classification.Classifier.Classifier import Classifier
from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.MultipleRun import MultipleRun
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Parameter.Parameter import Parameter
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class KFoldRun(MultipleRun):

    K: int

    def __init__(self, K: int):
        """
        Constructor for KFoldRun class. Basically sets K parameter of the K-fold cross-validation.

        PARAMETERS
        ----------
        K : int
            K of the K-fold cross-validation.
        """
        self.K = K

    def runExperiment(self, classifier: Classifier, parameter: Parameter, experimentPerformance: ExperimentPerformance,
                      crossValidation: CrossValidation):
        for i in range(self.K):
            trainSet = InstanceList(crossValidation.getTrainFold(i))
            testSet = InstanceList(crossValidation.getTestFold(i))
            classifier.train(trainSet, parameter)
            experimentPerformance.add(classifier.test(testSet))

    def execute(self, experiment: Experiment) -> ExperimentPerformance:
        """
        Execute K-fold cross-validation with the given classifier on the given data set using the given parameters.

        PARAMETERS
        ----------
        experiment : Experiment
            Experiment to be run.

        RETURNS
        -------
        ExperimentPerformance
            An ExperimentPerformance instance.
        """
        result = ExperimentPerformance()
        crossValidation = KFoldCrossValidation(experiment.getDataSet().getInstances(), self.K, experiment.getParameter()
                                               .getSeed())
        self.runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation)
        return result

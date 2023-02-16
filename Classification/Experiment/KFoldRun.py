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

    def runExperiment(self,
                      classifier: Classifier,
                      parameter: Parameter,
                      experimentPerformance: ExperimentPerformance,
                      crossValidation: CrossValidation):
        for i in range(self.K):
            train_set = InstanceList(crossValidation.getTrainFold(i))
            test_set = InstanceList(crossValidation.getTestFold(i))
            classifier.train(train_set, parameter)
            experimentPerformance.add(classifier.test(test_set))

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
        crossValidation = KFoldCrossValidation(instance_list=experiment.getDataSet().getInstances(),
                                               K=self.K,
                                               seed=experiment.getParameter().getSeed())
        self.runExperiment(classifier=experiment.getClassifier(),
                           parameter=experiment.getParameter(),
                           experimentPerformance=result,
                           crossValidation=crossValidation)
        return result

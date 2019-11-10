from Sampling.StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation

from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.StratifiedKFoldRunSeparateTest import StratifiedKFoldRunSeparateTest
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class StratifiedMxKFoldRunSeparateTest(StratifiedKFoldRunSeparateTest):

    M: int

    """
    Constructor for StratifiedMxKFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation
    and M for the number of times.

    PARAMETERS
    ----------
    M : int
        number of cross-validation times.
    K : int
        K of the K-fold cross-validation.
    """
    def __init__(self, M: int, K: int):
        super().__init__(K)
        self.M = M

    def execute(self, experiment: Experiment) -> ExperimentPerformance:
        result = ExperimentPerformance()
        for j in range(self.M):
            instanceList = experiment.getDataSet().getInstanceList()
            partition = instanceList.partition(0.25, experiment.getParameter().getSeed())
            crossValidation = StratifiedKFoldCrossValidation(partition.get(1).divideIntoClasses().getLists(), self.K, experiment.getParameter().getSeed())
            self.runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation, partition.get(0))
        return result

from Sampling.StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation

from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRunSeparateTest import KFoldRunSeparateTest
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class StratifiedKFoldRunSeparateTest(KFoldRunSeparateTest):

    """
    Constructor for StratifiedKFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation.

    PARAMETERS
    ----------
    K : int
        K of the K-fold cross-validation.
    """
    def __init__(self, K: int):
        super().__init__(K)

    """
    Execute Stratified K-fold cross-validation with the given classifier on the given data set using the given parameters.

    PARAMETERS
    ----------
    experiment : Experiment
        Experiment to be run.
        
    RETURNS
    -------
    ExperimentPerformance
        An ExperimentPerformance instance.
    """
    def execute(self, experiment: Experiment) -> ExperimentPerformance:
        result = ExperimentPerformance()
        instanceList = experiment.getDataSet().getInstanceList()
        partition = instanceList.partition(0.25, experiment.getParameter().getSeed())
        crossValidation = StratifiedKFoldCrossValidation(partition.get(1).divideIntoClasses().getLists(), self.K, experiment.getParameter().getSeed())
        self.runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation, partition.get(0))
        return result
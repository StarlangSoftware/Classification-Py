from Sampling.StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation

from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRun import KFoldRun
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class StratifiedKFoldRun(KFoldRun):

    """
    Constructor for KFoldRun class. Basically sets K parameter of the K-fold cross-validation.

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
        crossValidation = StratifiedKFoldCrossValidation(experiment.getDataSet().getClassInstances(), self.K, experiment.getParameter().getSeed())
        self.runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation)
        return result

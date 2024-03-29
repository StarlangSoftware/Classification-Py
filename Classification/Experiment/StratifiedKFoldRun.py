from Sampling.StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation

from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRun import KFoldRun
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class StratifiedKFoldRun(KFoldRun):

    def __init__(self, K: int):
        """
        Constructor for KFoldRun class. Basically sets K parameter of the K-fold cross-validation.

        PARAMETERS
        ----------
        K : int
            K of the K-fold cross-validation.
        """
        super().__init__(K)

    def execute(self, experiment: Experiment) -> ExperimentPerformance:
        """
        Execute Stratified K-fold cross-validation with the given classifier on the given data set using the given
        parameters.

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
        cross_validation = StratifiedKFoldCrossValidation(instance_lists=experiment.getDataSet().getClassInstances(),
                                                         K=self.K,
                                                         seed=experiment.getParameter().getSeed())
        self.runExperiment(classifier=experiment.getClassifier(),
                           parameter=experiment.getParameter(),
                           experimentPerformance=result,
                           crossValidation=cross_validation)
        return result

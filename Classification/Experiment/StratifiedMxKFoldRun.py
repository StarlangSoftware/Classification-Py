from Sampling.StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation

from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.MxKFoldRun import MxKFoldRun
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class StratifiedMxKFoldRun(MxKFoldRun):

    def __init__(self,
                 M: int,
                 K: int):
        """
        Constructor for StratifiedMxKFoldRun class. Basically sets K parameter of the K-fold cross-validation and M for
        the number of times.

        PARAMETERS
        ----------
        M : int
            number of cross-validation times.
        K : int
            K of the K-fold cross-validation.
        """
        super().__init__(M, K)

    def execute(self, experiment: Experiment) -> ExperimentPerformance:
        """
        Execute the Stratified MxK-fold cross-validation with the given classifier on the given data set using the given
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
        for j in range(self.M):
            cross_validation = StratifiedKFoldCrossValidation(instance_lists=experiment.getDataSet().getClassInstances(),
                                                             K=self.K,
                                                             seed=experiment.getParameter().getSeed())
            self.runExperiment(classifier=experiment.getClassifier(),
                               parameter=experiment.getParameter(),
                               experimentPerformance=result,
                               crossValidation=cross_validation)
        return result

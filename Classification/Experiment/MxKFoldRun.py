from Sampling.KFoldCrossValidation import KFoldCrossValidation

from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRun import KFoldRun
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class MxKFoldRun(KFoldRun):

    M: int

    def __init__(self, M: int, K: int):
        """
        Constructor for MxKFoldRun class. Basically sets K parameter of the K-fold cross-validation and M for the number
        of times.

        PARAMETERS
        ----------
        M : int
            number of cross-validation times.
        K : int
            K of the K-fold cross-validation.
        """
        super().__init__(K)
        self.M = M

    def execute(self, experiment: Experiment) -> ExperimentPerformance:
        """
        Execute the MxKFold run with the given classifier on the given data set using the given parameters.

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
            crossValidation = KFoldCrossValidation(experiment.getDataSet().getInstances(), self.K,
                                                   experiment.getParameter().getSeed())
            self.runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation)
        return result

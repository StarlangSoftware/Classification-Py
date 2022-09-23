from Sampling.KFoldCrossValidation import KFoldCrossValidation

from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRunSeparateTest import KFoldRunSeparateTest
from Classification.InstanceList.Partition import Partition
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class MxKFoldRunSeparateTest(KFoldRunSeparateTest):

    M: int

    def __init__(self,
                 M: int,
                 K: int):
        """
        Constructor for KFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation and M for
        the number of times.

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
        Execute the MxKFold run with separate test set with the given classifier on the given data set using the given
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
        instance_list = experiment.getDataSet().getInstanceList()
        partition = Partition(instanceList=instance_list,
                              ratio=0.25,
                              seed=experiment.getParameter().getSeed(),
                              stratified=True)
        for j in range(self.M):
            cross_validation = KFoldCrossValidation(instanceList=partition.get(1).getInstances(),
                                                   K=self.K,
                                                   seed=experiment.getParameter().getSeed())
            self.runExperimentSeparate(classifier=experiment.getClassifier(),
                               parameter=experiment.getParameter(),
                               experimentPerformance=result,
                               crossValidation=cross_validation,
                               testSet=partition.get(0))
        return result

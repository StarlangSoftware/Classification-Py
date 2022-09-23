from Sampling.StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation

from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.StratifiedKFoldRunSeparateTest import StratifiedKFoldRunSeparateTest
from Classification.InstanceList.Partition import Partition
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class StratifiedMxKFoldRunSeparateTest(StratifiedKFoldRunSeparateTest):

    M: int

    def __init__(self,
                 M: int,
                 K: int):
        """
        Constructor for StratifiedMxKFoldRunSeparateTest class. Basically sets K parameter of the K-fold
        cross-validation and M for the number of times.

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
        result = ExperimentPerformance()
        for j in range(self.M):
            instance_list = experiment.getDataSet().getInstanceList()
            partition = Partition(instanceList=instance_list,
                                  ratio=0.25,
                                  seed=experiment.getParameter().getSeed(),
                                  stratified=True)
            cross_validation = StratifiedKFoldCrossValidation(instanceLists=Partition(partition.get(1)).getLists(),
                                                             K=self.K,
                                                             seed=experiment.getParameter().getSeed())
            self.runExperimentSeparate(classifier=experiment.getClassifier(),
                               parameter=experiment.getParameter(),
                               experimentPerformance=result,
                               crossValidation=cross_validation,
                               testSet=partition.get(0))
        return result

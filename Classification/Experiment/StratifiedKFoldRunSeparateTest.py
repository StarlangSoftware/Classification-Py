from Sampling.StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation

from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRunSeparateTest import KFoldRunSeparateTest
from Classification.InstanceList.Partition import Partition
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class StratifiedKFoldRunSeparateTest(KFoldRunSeparateTest):

    def __init__(self, K: int):
        """
        Constructor for StratifiedKFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation.

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

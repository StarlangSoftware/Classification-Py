from Sampling.StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation

from Classification.Experiment.Experiment import Experiment
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Performance.Performance import Performance


class StratifiedSingleRunWithK:

    __K: int

    def __init__(self, K: int):
        """
        Constructor for StratifiedSingleRunWithK class. Basically sets K parameter of the K-fold cross-validation.

        PARAMETERS
        ----------
        K : int
            K of the K-fold cross-validation.
        """
        self.__K = K

    def execute(self, experiment: Experiment) -> Performance:
        """
        Execute Stratified Single K-fold cross-validation with the given classifier on the given data set using the
        given parameters.

        PARAMETERS
        ----------
        experiment : Experiment
            Experiment to be run.

        RETURNS
        -------
        Performance
            A Performance instance.
        """
        cross_validation = StratifiedKFoldCrossValidation(instanceLists=experiment.getDataSet().getClassInstances(),
                                                         K=self.__K,
                                                         seed=experiment.getParameter().getSeed())
        train_set = InstanceList(cross_validation.getTrainFold(0))
        test_set = InstanceList(cross_validation.getTestFold(0))
        return experiment.getClassifier().singleRun(parameter=experiment.getParameter(),
                                                    trainSet=train_set,
                                                    testSet=test_set)

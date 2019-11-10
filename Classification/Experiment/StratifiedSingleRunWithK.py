from Sampling.StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation

from Classification.Experiment.Experiment import Experiment
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Performance.Performance import Performance


class StratifiedSingleRunWithK:

    __K: int

    """
    Constructor for StratifiedSingleRunWithK class. Basically sets K parameter of the K-fold cross-validation.

    PARAMETERS
    ----------
    K : int
        K of the K-fold cross-validation.
    """
    def __init__(self, K: int):
        self.__K = K

    """
    Execute Stratified Single K-fold cross-validation with the given classifier on the given data set using the given parameters.

    PARAMETERS
    ----------
    experiment : Experiment
        Experiment to be run.
        
    RETURNS
    -------
    Performance
        A Performance instance.
    """
    def execute(self, experiment: Experiment) -> Performance:
        crossValidation = StratifiedKFoldCrossValidation(experiment.getDataSet().getClassInstances(), self.__K, experiment.getParameter().getSeed())
        trainSet = InstanceList(crossValidation.getTrainFold(0))
        testSet = InstanceList(crossValidation.getTestFold(0))
        return experiment.getClassifier().singleRun(experiment.getParameter(), trainSet, testSet)

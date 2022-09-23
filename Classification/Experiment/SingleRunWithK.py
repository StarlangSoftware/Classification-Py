from Sampling.CrossValidation import CrossValidation
from Sampling.KFoldCrossValidation import KFoldCrossValidation

from Classification.Classifier.Classifier import Classifier
from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.SingleRun import SingleRun
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Parameter.Parameter import Parameter
from Classification.Performance.Performance import Performance


class SingleRunWithK(SingleRun):

    __K: int

    def __init__(self, K: int):
        """
        Constructor for SingleRunWithK class. Basically sets K parameter of the K-fold cross-validation.

        PARAMETERS
        ----------
        K : int
            K of the K-fold cross-validation.
        """
        self.__K = K

    def runExperiment(self,
                      classifier: Classifier,
                      parameter: Parameter,
                      crossValidation: CrossValidation):
        train_set = InstanceList(crossValidation.getTrainFold(0))
        test_set = InstanceList(crossValidation.getTestFold(0))
        return classifier.singleRun(parameter=parameter,
                                    trainSet=train_set,
                                    testSet=test_set)

    def execute(self, experiment: Experiment) -> Performance:
        """
        Execute Single K-fold cross-validation with the given classifier on the given data set using the given
        parameters.

        PARAMETERS
        -----
        experiment : Experiment
            Experiment to be run.

        RETURNS
        -------
        Performance
            A Performance instance.
        """
        cross_validation = KFoldCrossValidation(instanceList=experiment.getDataSet().getInstances(),
                                               K=self.__K,
                                               seed=experiment.getParameter().getSeed())
        return self.runExperiment(classifier=experiment.getClassifier(),
                                  parameter=experiment.getParameter(),
                                  crossValidation=cross_validation)

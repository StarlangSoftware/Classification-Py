from Sampling.CrossValidation import CrossValidation
from Sampling.KFoldCrossValidation import KFoldCrossValidation

from Classification.Classifier.Classifier import Classifier
from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRun import KFoldRun
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Parameter.Parameter import Parameter
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class KFoldRunSeparateTest(KFoldRun):

    def __init__(self, K: int):
        """
        Constructor for KFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation.

        PARAMETERS
        ----------
        K : int
            K of the K-fold cross-validation.
        """
        super().__init__(K)

    def runExperimentSeparate(self,
                      classifier: Classifier,
                      parameter: Parameter,
                      experimentPerformance: ExperimentPerformance,
                      crossValidation: CrossValidation,
                      testSet: InstanceList):
        for i in range(self.K):
            train_set = InstanceList(crossValidation.getTrainFold(i))
            classifier.train(train_set, parameter)
            experimentPerformance.add(classifier.test(testSet))

    def execute(self, experiment: Experiment) -> ExperimentPerformance:
        """
        Execute K-fold cross-validation with separate test set with the given classifier on the given data set using the
        given parameters.

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
        cross_validation = KFoldCrossValidation(instanceList=partition.get(1).getInstances(),
                                               K=self.K,
                                               seed=experiment.getParameter().getSeed())
        self.runExperimentSeparate(classifier=experiment.getClassifier(),
                           parameter=experiment.getParameter(),
                           experimentPerformance=result,
                           crossValidation=cross_validation,
                           testSet=partition.get(0))
        return result

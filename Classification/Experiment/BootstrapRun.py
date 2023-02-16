from Classification.Experiment.MultipleRun import MultipleRun
from Classification.Experiment.Experiment import Experiment
from Classification.Performance.ExperimentPerformance import ExperimentPerformance
from Sampling.Bootstrap import Bootstrap
from Classification.InstanceList.InstanceList import InstanceList


class BootstrapRun(MultipleRun):

    __numberOfBootstraps: int

    def __init__(self, numberOfBootstraps: int):
        """
        Constructor for BootstrapRun class. Basically sets the number of bootstrap runs.

        PARAMETERS
        ----------
        numberOfBootstraps : int
            Number of bootstrap runs.
        """
        self.__numberOfBootstraps = numberOfBootstraps

    def execute(self, experiment: Experiment) -> ExperimentPerformance:
        """
        Execute the bootstrap run with the given classifier on the given data set using the given parameters.

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
        for i in range(self.__numberOfBootstraps):
            bootstrap = Bootstrap(instance_list=experiment.getDataSet().getInstances(),
                                  seed=i + experiment.getParameter().getSeed())
            bootstrap_sample = InstanceList(bootstrap.getSample())
            experiment.getClassifier().train(trainSet=bootstrap_sample,
                                             parameters=experiment.getParameter())
            result.add(experiment.getClassifier().test(experiment.getDataSet().getInstanceList()))
        return result

from abc import abstractmethod
from Classification.Experiment.Experiment import Experiment
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class MultipleRun(object):

    @abstractmethod
    def execute(self, experiment: Experiment) -> ExperimentPerformance:
        pass

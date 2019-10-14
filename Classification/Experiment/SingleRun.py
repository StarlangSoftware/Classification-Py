from abc import abstractmethod
from Classification.Experiment.Experiment import Experiment
from Classification.Performance.Performance import Performance


class SingleRun(object):

    @abstractmethod
    def execute(self, experiment: Experiment) -> Performance:
        pass

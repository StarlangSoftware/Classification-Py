from abc import abstractmethod
from io import TextIOWrapper

from Math.DiscreteDistribution import DiscreteDistribution
from Math.Vector import Vector

from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.Model.ValidatedModel import ValidatedModel


class GaussianModel(ValidatedModel):

    prior_distribution: DiscreteDistribution

    @abstractmethod
    def calculateMetric(self,
                        instance: Instance,
                        Ci: str) -> float:
        pass

    def loadPriorDistribution(self, inputFile: TextIOWrapper):
        """
        Loads the prior probability distribution from an input model file.
        :param inputFile: Input model file.
        :return: Prior probability distribution.
        """
        size = int(inputFile.readline().strip())
        self.prior_distribution = DiscreteDistribution()
        for i in range(size):
            line = inputFile.readline().strip()
            items = line.split(" ")
            for j in range(int(items[1])):
                self.prior_distribution.addItem(items[0])
        return size

    def loadVectors(self,
                    inputFile: TextIOWrapper,
                    size: int) -> dict:
        """
        Loads hash map of vectors from input model file.
        :param inputFile: Input model file.
        :param size: Number of vectors to be read from input model file.
        :return: Dictionary of vectors.
        """
        hash_map = dict()
        for i in range(size):
            line = inputFile.readline().strip()
            items = line.split(" ")
            vector = Vector(int(items[1]), 0)
            for j in range(2, len(items)):
                vector.setValue(j - 2, float(items[j]))
            hash_map[items[0]] = vector
        return hash_map

    def predict(self, instance: Instance) -> str:
        """
        The predict method takes an Instance as an input. First it gets the size of prior distribution and loops this
        size times. Then it gets the possible class labels and and calculates metric value. At the end, it returns the
        class which has the maximum value of metric.

        PARAMETERS
        ----------
        instance : Instance
            Instance to predict.

        RETURNS
        -------
        str
            The class which has the maximum value of metric.
        """
        max_metric = -10000000
        if isinstance(instance, CompositeInstance):
            predicted_class = instance.getPossibleClassLabels()[0]
            size = len(instance.getPossibleClassLabels())
        else:
            predicted_class = self.prior_distribution.getMaxItem()
            size = len(self.prior_distribution)
        for i in range(size):
            if isinstance(instance, CompositeInstance):
                Ci = instance.getPossibleClassLabels()[i]
            else:
                Ci = self.prior_distribution.getItem(i)
            if self.prior_distribution.containsItem(Ci):
                metric = self.calculateMetric(instance, Ci)
                if metric > max_metric:
                    max_metric = metric
                    predicted_class = Ci
        return predicted_class
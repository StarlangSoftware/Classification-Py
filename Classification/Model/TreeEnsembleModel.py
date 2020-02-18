from Math.DiscreteDistribution import DiscreteDistribution

from Classification.Instance.Instance import Instance
from Classification.Model.Model import Model


class TreeEnsembleModel(Model):

    __forest: list

    def __init__(self, forest: list):
        """
        A constructor which sets the list of DecisionTree with given input.

        PARAMETERS
        ----------
        forest list
            A list of DecisionTrees.
        """
        self.__forest = forest

    def predict(self, instance: Instance) -> str:
        """
        The predict method takes an Instance as an input and loops through the list of DecisionTrees.
        Makes prediction for the items of that ArrayList and returns the maximum item of that ArrayList.

        PARAMETERS
        ----------
        instance : Instance
            Instance to make prediction.

        RETURNS
        -------
        str
            The maximum prediction of a given Instance.
        """
        distribution = DiscreteDistribution()
        for tree in self.__forest:
            distribution.addItem(tree.predict(instance))
        return distribution.getMaxItem()

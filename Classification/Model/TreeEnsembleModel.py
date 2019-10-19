from Math.DiscreteDistribution import DiscreteDistribution

from Classification.Instance.Instance import Instance
from Classification.Model.Model import Model


class TreeEnsembleModel(Model):

    """
    A constructor which sets the list of DecisionTree with given input.

    PARAMETERS
    ----------
    forest list
        A list of DecisionTrees.
    """
    def __init__(self, forest: list):
        self.forest = forest

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
    def predict(self, instance: Instance) -> str:
        distribution = DiscreteDistribution()
        for tree in self.forest:
            distribution.addItem(tree.predict(instance))
        return distribution.getMaxItem()

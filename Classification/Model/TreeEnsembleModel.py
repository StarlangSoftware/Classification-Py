from Math.DiscreteDistribution import DiscreteDistribution

from Classification.Instance.Instance import Instance
from Classification.Model.DecisionTree.DecisionNode import DecisionNode
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Model.Model import Model


class TreeEnsembleModel(Model):

    __forest: list

    def constructor1(self, forest: list):
        """
        A constructor which sets the list of DecisionTree with given input.

        PARAMETERS
        ----------
        forest list
            A list of DecisionTrees.
        """
        self.__forest = forest

    def constructor2(self, fileName: str):
        inputFile = open(fileName, mode='r', encoding='utf-8')
        number_of_trees = int(inputFile.readline().strip())
        self.__forest = list()
        for i in range(number_of_trees):
            self.__forest.append(DecisionTree(DecisionNode(inputFile)))
        inputFile.close()

    def __init__(self, forest: object):
        if isinstance(forest, list):
            self.constructor1(forest)
        elif isinstance(forest, str):
            self.constructor2(forest)

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

    def predictProbability(self, instance: Instance) -> dict:
        distribution = DiscreteDistribution()
        for tree in self.__forest:
            distribution.addItem(tree.predict(instance))
        return distribution.getProbabilityDistribution()

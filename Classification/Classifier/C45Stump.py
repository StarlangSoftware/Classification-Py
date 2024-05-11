from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.DecisionNode import DecisionNode
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Parameter.Parameter import Parameter


class C45Stump(Classifier):

    def train(self,
              trainSet: InstanceList,
              parameters: Parameter):
        """
        Training algorithm for C4.5 Stump univariate decision tree classifier.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        parameters: Parameter
            Parameter of the C45Stump algorithm.
        """
        self.model = DecisionTree(DecisionNode(data=trainSet,
                                               isStump=True))

    def loadModel(self, fileName: str):
        """
        Loads the decision tree model from an input file.
        :param fileName: File name of the decision tree model.
        """
        self.model = DecisionTree(fileName)

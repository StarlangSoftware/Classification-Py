from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.DecisionNode import DecisionNode
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Model.TreeEnsembleModel import TreeEnsembleModel
from Classification.Parameter.RandomForestParameter import RandomForestParameter


class RandomForest(Classifier):

    """
    Training algorithm for random forest classifier. Basically the algorithm creates K distinct decision trees from
    K bootstrap samples of the original training set.

    PARAMETERS
    ----------
    trainSet : InstanceList
        Training data given to the algorithm
    parameters : RandomForestParameter
        Parameters of the bagging trees algorithm. ensembleSize returns the number of trees in the random forest.
    """
    def train(self, trainSet: InstanceList, parameters: RandomForestParameter):
        forestSize = parameters.getEnsembleSize()
        forest = []
        for i in range(forestSize):
            bootstrap = trainSet.bootstrap(i)
            forest.append(DecisionTree(DecisionNode(InstanceList(bootstrap.getSample(), None, parameters, False))))
        self.model = TreeEnsembleModel(forest)
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.DecisionNode import DecisionNode
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Model.TreeEnsembleModel import TreeEnsembleModel
from Classification.Parameter.RandomForestParameter import RandomForestParameter


class RandomForestModel(TreeEnsembleModel):

    def train(self,
              trainSet: InstanceList,
              parameters: RandomForestParameter):
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
        forest_size = parameters.getEnsembleSize()
        forest = []
        for i in range(forest_size):
            bootstrap = trainSet.bootstrap(i)
            tree = DecisionTree(DecisionNode(data=InstanceList(bootstrap.getSample()),
                                             parameter=parameters,
                                             isStump=False))
            forest.append(tree)
        self.constructor1(forest)

    def loadModel(self, fileName: str):
        """
        Loads the random forest model from an input file.
        :param fileName: File name of the random forest model.
        """
        self.constructor2(fileName)

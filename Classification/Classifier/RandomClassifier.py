from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.RandomModel import RandomModel
from Classification.Parameter.Parameter import Parameter


class RandomClassifier(Classifier):

    def train(self, trainSet: InstanceList, parameters: Parameter):
        """
        Training algorithm for random classifier.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        """
        self.model = RandomModel(list(trainSet.classDistribution().keys()), parameters.getSeed())

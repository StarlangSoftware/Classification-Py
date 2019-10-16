from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.RandomModel import RandomModel
from Classification.Parameter.Parameter import Parameter


class RandomClassifier(Classifier):

    """
    Training algorithm for random classifier.

    PARAMETERS
    ----------
    trainSet : InstanceList
        Training data given to the algorithm.
    """
    def train(self, trainSet: InstanceList, parameters: Parameter):
        self.model = RandomModel(list(trainSet.classDistribution().keys()))

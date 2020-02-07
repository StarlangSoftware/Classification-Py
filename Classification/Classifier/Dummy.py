from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DummyModel import DummyModel
from Classification.Parameter.Parameter import Parameter


class Dummy(Classifier):

    """
    Training algorithm for the dummy classifier. Actually dummy classifier returns the maximum occurring class in
    the training data, there is no training.

    PARAMETERS
    ----------
    trainSet: InstanceList
        Training data given to the algorithm.
    """
    def train(self, trainSet: InstanceList, parameters: Parameter):
        self.model = DummyModel(trainSet)

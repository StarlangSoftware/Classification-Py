from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DummyModel import DummyModel
from Classification.Parameter.Parameter import Parameter


class Dummy(Classifier):

    def train(self, trainSet: InstanceList, parameters: Parameter):
        """
        Training algorithm for the dummy classifier. Actually dummy classifier returns the maximum occurring class in
        the training data, there is no training.

        PARAMETERS
        ----------
        trainSet: InstanceList
            Training data given to the algorithm.
        parameters: Parameter
            Parameter of the Dummy algorithm.
        """
        self.model = DummyModel(trainSet)

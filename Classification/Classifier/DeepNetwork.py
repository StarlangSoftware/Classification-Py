from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DeepNetworkModel import DeepNetworkModel
from Classification.Parameter.DeepNetworkParameter import DeepNetworkParameter


class DeepNetwork(Classifier):

    """
    Training algorithm for deep network classifier.

    PARAMETERS
    ----------
    trainSet : InstanceList
        Training data given to the algorithm.
    parameters : DeepNetworkParameter
        Parameters of the deep network algorithm. crossValidationRatio and seed are used as parameters.
    """
    def train(self, trainSet: InstanceList, parameters: DeepNetworkParameter):
        partition = trainSet.stratifiedPartition(parameters.getCrossValidationRatio(), parameters.getSeed())
        self.model = DeepNetworkModel(partition.get(1), partition.get(0), parameters)

from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.DeepNetworkModel import DeepNetworkModel
from Classification.Parameter.DeepNetworkParameter import DeepNetworkParameter


class DeepNetwork(Classifier):

    def train(self, trainSet: InstanceList, parameters: DeepNetworkParameter):
        """
        Training algorithm for deep network classifier.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        parameters : DeepNetworkParameter
            Parameters of the deep network algorithm. crossValidationRatio and seed are used as parameters.
        """
        partition = Partition(trainSet, parameters.getCrossValidationRatio(), parameters.getSeed(), True)
        self.model = DeepNetworkModel(partition.get(1), partition.get(0), parameters)

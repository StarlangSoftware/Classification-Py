from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.MultiLayerPerceptronModel import MultiLayerPerceptronModel
from Classification.Parameter.MultiLayerPerceptronParameter import MultiLayerPerceptronParameter


class MultiLayerPerceptron(Classifier):

    def train(self, trainSet: InstanceList, parameters: MultiLayerPerceptronParameter):
        """
        Training algorithm for the multilayer perceptron algorithm. 20 percent of the data is separated as
        cross-validation data used for selecting the best weights. 80 percent of the data is used for training the
        multilayer perceptron with gradient descent.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm
        parameters : MultiLayerPerceptronParameter
            Parameters of the multilayer perceptron.
        """
        partition = Partition(trainSet, parameters.getCrossValidationRatio(), parameters.getSeed(), True)
        self.model = MultiLayerPerceptronModel(partition.get(1), partition.get(0), parameters)

from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.KnnModel import KnnModel
from Classification.Parameter.KnnParameter import KnnParameter


class Knn(Classifier):

    def train(self, trainSet: InstanceList, parameters: KnnParameter):
        """
        Training algorithm for K-nearest neighbor classifier.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        parameters : KnnParameter
            Parameters of the Knn algorithm.
        """
        self.model = KnnModel(trainSet, parameters.getK(), parameters.getDistanceMetric())

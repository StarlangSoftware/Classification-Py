from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.KnnModel import KnnModel
from Classification.Parameter.KnnParameter import KnnParameter


class Knn(Classifier):

    """
    Training algorithm for K-nearest neighbor classifier.

    PARAMETERS
    ----------
    trainSet : InstanceList
        Training data given to the algorithm.
    parameters : KnnParameter
        K: k parameter of the K-nearest neighbor algorithm
        distanceMetric: distance metric used to calculate the distance between two instances.
    """
    def train(self, trainSet: InstanceList, parameters: KnnParameter):
        self.model = KnnModel(trainSet, parameters.getK(), parameters.getDistanceMetric())

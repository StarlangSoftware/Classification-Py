from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.KnnModel import KnnModel
from Classification.Parameter.KnnParameter import KnnParameter


class Knn(Classifier):

    def train(self,
              trainSet: InstanceList,
              parameters: KnnParameter):
        """
        Training algorithm for K-nearest neighbor classifier.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        parameters : KnnParameter
            Parameters of the Knn algorithm.
        """
        self.model = KnnModel(data=trainSet,
                              k=parameters.getK(),
                              distanceMetric=parameters.getDistanceMetric())

    def loadModel(self, fileName: str):
        """
        Loads the K-nearest neighbor model from an input file.
        :param fileName: File name of the K-nearest neighbor model.
        """
        self.model = KnnModel(fileName)

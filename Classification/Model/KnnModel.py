from functools import cmp_to_key
from io import TextIOWrapper

from Classification.DistanceMetric.DistanceMetric import DistanceMetric
from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.KnnInstance import KnnInstance
from Classification.Model.Model import Model


class KnnModel(Model):
    __data: InstanceList
    __k: int
    __distance_metric: DistanceMetric

    def constructor1(self,
                     data: InstanceList,
                     k: int,
                     distanceMetric: DistanceMetric):
        """
        Constructor that sets the data InstanceList, k value and the DistanceMetric.

        PARAMETERS
        ----------
        data : InstanceList
            InstanceList input.
        k : int
            K value.
        distanceMetric : DistanceMetric
            DistanceMetric input.
        """
        self.__data = data
        self.__k = k
        self.__distance_metric = distanceMetric

    def constructor2(self, fileName: str):
        """
        Loads a K-nearest neighbor model from an input model file.
        :param fileName: Model file name.
        """
        self.__distance_metric = EuclidianDistance()
        inputFile = open(fileName, 'r')
        self.__k = int(inputFile.readline().strip())
        self.__data = self.loadInstanceList(inputFile)
        inputFile.close()

    def __init__(self,
                 data: object,
                 k: int = None,
                 distanceMetric: DistanceMetric = None):
        if isinstance(data, InstanceList):
            self.constructor1(data, k, distanceMetric)
        elif isinstance(data, str):
            self.constructor2(data)

    def predict(self, instance: Instance) -> str:
        """
        The predict method takes an Instance as an input and finds the nearest neighbors of given instance. Then
        it returns the first possible class label as the predicted class.

        PARAMETERS
        ----------
        instance : Instance
            Instance to make prediction.

        RETURNS
        -------
        str
            The first possible class label as the predicted class.
        """
        nearest_neighbors = self.nearestNeighbors(instance)
        if isinstance(instance, CompositeInstance) and nearest_neighbors.size() == 0:
            predicted_class = instance.getPossibleClassLabels()[0]
        else:
            predicted_class = Model.getMaximum(nearest_neighbors.getClassLabels())
        return predicted_class

    def predictProbability(self, instance: Instance) -> dict:
        """
        Calculates the posterior probability distribution for the given instance according to K-means model.
        :param instance: Instance for which posterior probability distribution is calculated.
        :return: Posterior probability distribution for the given instance.
        """
        nearest_neighbors = self.nearestNeighbors(instance)
        return nearest_neighbors.classDistribution().getProbabilityDistribution()

    def makeComparator(self):
        def compare(instanceA: KnnInstance, instanceB: KnnInstance):
            if instanceA.distance < instanceB.distance:
                return -1
            elif instanceA.distance > instanceB.distance:
                return 1
            else:
                return 0

        return compare

    def nearestNeighbors(self, instance: Instance) -> InstanceList:
        """
        The nearestNeighbors method takes an Instance as an input. First it gets the possible class labels, then loops
        through the data InstanceList and creates new list of KnnInstances and adds the corresponding data with
        the distance between data and given instance. After sorting this newly created list, it loops k times and
        returns the first k instances as an InstanceList.

        PARAMETERS
        ----------
        instance : Instance
            Instance to find nearest neighbors

        RETURNS
        -------
        InstanceList
            The first k instances which are nearest to the given instance as an InstanceList.
        """
        result = InstanceList()
        instances = []
        possible_class_labels = []
        if isinstance(instance, CompositeInstance):
            possible_class_labels = instance.getPossibleClassLabels()
        for i in range(self.__data.size()):
            if not isinstance(instance, CompositeInstance) or self.__data.get(
                    i).getClassLabel() in possible_class_labels:
                instances.append(KnnInstance(self.__data.get(i), self.__distance_metric.distance(self.__data.get(i),
                                                                                                 instance)))
        instances.sort(key=cmp_to_key(self.makeComparator()))
        for i in range(min(self.__k, len(instances))):
            result.add(instances[i].instance)
        return result

from io import TextIOWrapper

from Math.DiscreteDistribution import DiscreteDistribution

from Classification.DistanceMetric.DistanceMetric import DistanceMetric
from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.GaussianModel import GaussianModel
from Classification.Parameter.KMeansParameter import KMeansParameter


class KMeansModel(GaussianModel):
    __class_means: InstanceList
    __distance_metric: DistanceMetric

    def constructor1(self,
                     priorDistribution: DiscreteDistribution,
                     classMeans: InstanceList,
                     distanceMetric: DistanceMetric):
        """
        The constructor that sets the classMeans, priorDistribution and distanceMetric according to given inputs.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            DiscreteDistribution input.
        classMeans : InstanceList
            Class means.
        distanceMetric : DistanceMetric
            DistanceMetric input.
        """
        self.__class_means = classMeans
        self.prior_distribution = priorDistribution
        self.__distance_metric = distanceMetric

    def constructor2(self, fileName: str):
        """
        Loads a K-means model from an input model file.
        :param fileName: Model file name.
        """
        self.__distance_metric = EuclidianDistance()
        inputFile = open(fileName, 'r')
        self.loadPriorDistribution(inputFile)
        self.__class_means = self.loadInstanceList(inputFile)
        inputFile.close()

    def __init__(self,
                 priorDistribution: object = None,
                 classMeans: InstanceList = None,
                 distanceMetric: DistanceMetric = None):
        if isinstance(priorDistribution, DiscreteDistribution):
            self.constructor1(priorDistribution, classMeans, distanceMetric)
        elif isinstance(priorDistribution, str):
            self.constructor2(priorDistribution)

    def calculateMetric(self,
                        instance: Instance,
                        Ci: str) -> float:
        """
        The calculateMetric method takes an {@link Instance} and a String as inputs. It loops through the class means,
        if the corresponding class label is same as the given String it returns the negated distance between given
        instance and the current item of class means. Otherwise it returns the smallest negative number.

        PARAMETERS
        ----------
        instance : Instance
            Instance input.
        Ci : str
            String input.

        RETURNS
        -------
        float
            The negated distance between given instance and the current item of class means.
        """
        for i in range(self.__class_means.size()):
            if self.__class_means.get(i).getClassLabel() == Ci:
                return -self.__distance_metric.distance(instance, self.__class_means.get(i))
        return -1000000

    def train(self,
              trainSet: InstanceList,
              parameters: KMeansParameter):
        """
        Training algorithm for K-Means classifier. K-Means finds the mean of each class for training.
        :param trainSet: Training data given to the algorithm.
        :param parameters: distance metric used to calculate the distance between two instances.
        """
        prior_distribution = trainSet.classDistribution()
        class_means = InstanceList()
        class_lists = Partition(trainSet)
        for i in range(class_lists.size()):
            class_means.add(class_lists.get(i).average())
        self.constructor1(priorDistribution=prior_distribution, classMeans=class_means, distanceMetric=parameters.getDistanceMetric())

    def loadModel(self, fileName: str):
        """
        Loads the K-means model from an input file.
        :param fileName: File name of the K-means model.
        """
        self.constructor2(fileName)

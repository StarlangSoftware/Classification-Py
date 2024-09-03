import math
from io import TextIOWrapper

from Math.DiscreteDistribution import DiscreteDistribution
from Math.Matrix import Matrix
from Math.Vector import Vector

from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.Parametric.GaussianModel import GaussianModel
from Classification.Parameter.Parameter import Parameter


class LdaModel(GaussianModel):
    w0: dict
    w: dict

    def constructor1(self,
                     priorDistribution: DiscreteDistribution,
                     w: dict,
                     w0: dict):
        """
        A constructor which sets the priorDistribution, w and w0 according to given inputs.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            DiscreteDistribution input.
        w : dict
            Dict of String and Vectors.
        w0 : dict
            Dict of String and float.
        """
        self.prior_distribution = priorDistribution
        self.w = w
        self.w0 = w0

    def constructor2(self, fileName: str):
        """
        Loads a Linear Discriminant Analysis model from an input model file.
        :param fileName: Model file name.
        """
        inputFile = open(fileName, mode='r', encoding='utf-8')
        size = self.loadPriorDistribution(inputFile)
        self.loadWandW0(inputFile, size)
        inputFile.close()

    def __init__(self,
                 priorDistribution: object = None,
                 w: dict = None,
                 w0: dict = None):
        if priorDistribution is not None:
            if isinstance(priorDistribution, DiscreteDistribution):
                self.constructor1(priorDistribution, w, w0)
            elif isinstance(priorDistribution, str):
                self.constructor2(priorDistribution)

    def loadWandW0(self,
                   inputFile: TextIOWrapper,
                   size: int):
        """
        Loads w0 and w hash maps from an input file. The number of items in the hash map is given by the parameter size.
        :param inputFile: Input file
        :param size: Size of the hash map
        """
        self.w0 = dict()
        for i in range(size):
            line = inputFile.readline().strip()
            items = line.split(" ")
            self.w0[items[0]] = float(items[1])
        self.w = self.loadVectors(inputFile, size)

    def calculateMetric(self,
                        instance: Instance,
                        Ci: str) -> float:
        """
        The calculateMetric method takes an Instance and a String as inputs. It returns the dot product of given
        Instance and wi plus w0i.

        PARAMETERS
        ----------
        instance : Instance
            Instance input.
        Ci : str
            String input.

        RETURNS
        -------
        float
            The dot product of given Instance and wi plus w0i.
        """
        xi = instance.toVector()
        wi = self.w[Ci]
        w0i = self.w0[Ci]
        return wi.dotProduct(xi) + w0i

    def train(self,
              trainSet: InstanceList,
              parameters: Parameter = None):
        """
        Training algorithm for the linear discriminant analysis classifier (Introduction to Machine Learning, Alpaydin,
        2015).

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        parameters : Parameter
            Parameter of the Lda algorithm.
        """
        w0 = {}
        w = {}
        prior_distribution = trainSet.classDistribution()
        class_lists = Partition(trainSet)
        covariance = Matrix(trainSet.get(0).continuousAttributeSize(), trainSet.get(0).continuousAttributeSize())
        for i in range(class_lists.size()):
            average_vector = Vector(class_lists.get(i).continuousAverage())
            class_covariance = class_lists.get(i).covariance(average_vector)
            class_covariance.multiplyWithConstant(class_lists.get(i).size() - 1)
            covariance.add(class_covariance)
        covariance.divideByConstant(trainSet.size() - class_lists.size())
        covariance.inverse()
        for i in range(class_lists.size()):
            Ci = class_lists.get(i).getClassLabel()
            average_vector = Vector(class_lists.get(i).continuousAverage())
            wi = covariance.multiplyWithVectorFromRight(average_vector)
            w[Ci] = wi
            w0i = -0.5 * wi.dotProduct(average_vector) + math.log(prior_distribution.getProbability(Ci))
            w0[Ci] = w0i
        self.constructor1(prior_distribution, w, w0)

    def loadModel(self, fileName: str):
        """
        Loads the Lda model from an input file.
        :param fileName: File name of the Lda model.
        """
        self.constructor2(fileName)

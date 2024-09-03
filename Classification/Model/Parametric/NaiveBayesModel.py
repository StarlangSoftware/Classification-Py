from Math.DiscreteDistribution import DiscreteDistribution

from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.Parametric.GaussianModel import GaussianModel
import math

from Classification.Parameter.Parameter import Parameter


class NaiveBayesModel(GaussianModel):

    __class_means: dict
    __class_deviations: dict
    __class_attribute_distributions: dict

    def constructor1(self, priorDistribution: DiscreteDistribution):
        """
        A constructor that sets the priorDistribution.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            DiscreteDistribution input.
        """
        self.prior_distribution = priorDistribution

    def constructor2(self, fileName: str):
        """
        Loads a naive Bayes model from an input model file.
        :param fileName: Model file name.
        """
        inputFile = open(fileName, mode='r', encoding='utf-8')
        size = self.loadPriorDistribution(inputFile)
        self.__class_means = self.loadVectors(inputFile, size)
        self.__class_deviations = self.loadVectors(inputFile, size)
        self.__class_attribute_distributions = None
        inputFile.close()

    def __init__(self, priorDistribution: object = None):
        if isinstance(priorDistribution, DiscreteDistribution):
            self.constructor1(priorDistribution)
        elif isinstance(priorDistribution, str):
            self.constructor2(priorDistribution)

    def initForContinuous(self,
                          classMeans: dict,
                          classDeviations: dict):
        """
        A constructor that sets the classMeans and classDeviations.

        PARAMETERS
        ----------
        classMeans : dict
            A dict of String and Vector.
        classDeviations : dict
            A dict of String and Vector.
        """
        self.__class_means = classMeans
        self.__class_deviations = classDeviations
        self.__class_attribute_distributions = None

    def initForDiscrete(self, classAttributeDistributions: dict):
        """
        A constructor that sets the priorDistribution and classAttributeDistributions.

        PARAMETERS
        ----------
        classAttributeDistributions : dict
            A dict of String and list of DiscreteDistributions.
        """
        self.__class_attribute_distributions = classAttributeDistributions

    def calculateMetric(self,
                        instance: Instance,
                        Ci: str) -> float:
        """
        The calculateMetric method takes an Instance and a String as inputs and it returns the log likelihood of
        these inputs.

        PARAMETERS
        ----------
        instance : Instance
            Instance input.
        Ci : str
            String input.

        RETURNS
        -------
        float
            The log likelihood of inputs.
        """
        if self.__class_attribute_distributions is None:
            return self.__logLikelihoodContinuous(Ci, instance)
        else:
            return self.__logLikelihoodDiscrete(Ci, instance)

    def __logLikelihoodContinuous(self,
                                  classLabel: str,
                                  instance: Instance) -> float:
        """
        The logLikelihoodContinuous method takes an Instance and a class label as inputs. First it gets the logarithm
        of given class label's probability via prior distribution as logLikelihood. Then it loops times of given
        instance attribute size, and accumulates the logLikelihood by calculating -0.5 * ((xi - mi) / si )** 2).

        PARAMETERS
        ----------
        classLabel : str
            String input class label.
        instance : Instance
            Instance input.

        RETURNS
        -------
        float
            The log likelihood of given class label and Instance.
        """
        log_likelihood = math.log(self.prior_distribution.getProbability(classLabel))
        for i in range(instance.attributeSize()):
            xi = instance.getAttribute(i).getValue()
            mi = self.__class_means[classLabel].getValue(i)
            si = self.__class_deviations[classLabel].getValue(i)
            if si != 0:
                log_likelihood += -0.5 * math.pow((xi - mi) / si, 2)
        return log_likelihood

    def __logLikelihoodDiscrete(self,
                                classLabel: str,
                                instance: Instance) -> float:
        """
        The logLikelihoodDiscrete method takes an Instance and a class label as inputs. First it gets the logarithm
        of given class label's probability via prior distribution as logLikelihood and gets the class attribute
        distribution of given class label. Then it loops times of given instance attribute size, and accumulates the
        logLikelihood by calculating the logarithm of corresponding attribute distribution's smoothed probability by
        using laplace smoothing on xi.

        PARAMETERS
        ----------
        classLabel : str
            String input class label.
        instance : Instance
            Instance input.

        RETURNS
        -------
        float
            The log likelihood of given class label and Instance.
        """
        log_likelihood = math.log(self.prior_distribution.getProbability(classLabel))
        attribute_distributions = self.__class_attribute_distributions.get(classLabel)
        for i in range(instance.attributeSize()):
            xi = instance.getAttribute(i).getValue()
            log_likelihood += math.log(attribute_distributions[i].getProbabilityLaplaceSmoothing(xi))
        return log_likelihood

    def trainContinuousVersion(self,
                               priorDistribution: DiscreteDistribution,
                               classLists: Partition):
        """
        Training algorithm for Naive Bayes algorithm with a continuous data set.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            Probability distribution of classes P(C_i)
        classLists : Partition
            Instances are divided into K lists, where each list contains only instances from a single class
        """
        class_means = {}
        class_deviations = {}
        for i in range(classLists.size()):
            class_label = classLists.get(i).getClassLabel()
            average_vector = classLists.get(i).average().toVector()
            class_means[class_label] = average_vector
            standard_deviation_vector = classLists.get(i).standardDeviation().toVector()
            class_deviations[class_label] = standard_deviation_vector
        self.constructor1(priorDistribution)
        self.initForContinuous(class_means, class_deviations)

    def trainDiscreteVersion(self,
                             priorDistribution: DiscreteDistribution,
                             classLists: Partition):
        """
        Training algorithm for Naive Bayes algorithm with a discrete data set.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            Probability distribution of classes P(C_i)
        classLists : Partition
            Instances are divided into K lists, where each list contains only instances from a single class
        """
        class_attribute_distributions = {}
        for i in range(classLists.size()):
            class_attribute_distributions[classLists.get(i).getClassLabel()] = \
                classLists.get(i).allAttributesDistribution()
        self.constructor1(priorDistribution)
        self.initForDiscrete(class_attribute_distributions)

    def train(self,
              trainSet: InstanceList,
              parameters: Parameter = None):
        """
        Training algorithm for Naive Bayes algorithm. It basically calls trainContinuousVersion for continuous data
        sets, trainDiscreteVersion for discrete data sets.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm
        """
        prior_distribution = trainSet.classDistribution()
        class_lists = Partition(trainSet)
        if isinstance(class_lists.get(0).get(0).getAttribute(0), DiscreteAttribute):
            self.trainDiscreteVersion(prior_distribution, class_lists)
        else:
            self.trainContinuousVersion(prior_distribution, class_lists)

    def loadModel(self, fileName: str):
        """
        Loads the naive Bayes model from an input file.
        :param fileName: File name of the naive Bayes model.
        """
        self.constructor2(fileName)

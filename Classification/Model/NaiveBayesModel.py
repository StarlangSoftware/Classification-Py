from Math.DiscreteDistribution import DiscreteDistribution

from Classification.Instance.Instance import Instance
from Classification.Model.GaussianModel import GaussianModel
import math


class NaiveBayesModel(GaussianModel):

    __classMeans: dict
    __classDeviations: dict
    __classAttributeDistributions: dict

    def __init__(self, priorDistribution: DiscreteDistribution):
        """
        A constructor that sets the priorDistribution.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            DiscreteDistribution input.
        """
        self.priorDistribution = priorDistribution

    def initForContinuous(self, classMeans: dict, classDeviations: dict):
        """
        A constructor that sets the classMeans and classDeviations.

        PARAMETERS
        ----------
        classMeans : dict
            A dict of String and Vector.
        classDeviations : dict
            A dict of String and Vector.
        """
        self.__classMeans = classMeans
        self.__classDeviations = classDeviations
        self.__classAttributeDistributions = None

    def initForDiscrete(self, classAttributeDistributions: dict):
        """
        A constructor that sets the priorDistribution and classAttributeDistributions.

        PARAMETERS
        ----------
        classAttributeDistributions : dict
            A dict of String and list of DiscreteDistributions.
        """
        self.__classAttributeDistributions = classAttributeDistributions

    def calculateMetric(self, instance: Instance, Ci: str) -> float:
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
        if self.__classAttributeDistributions is None:
            return self.__logLikelihoodContinuous(Ci, instance)
        else:
            return self.__logLikelihoodDiscrete(Ci, instance)

    def __logLikelihoodContinuous(self, classLabel: str, instance: Instance) -> float:
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
        loglikelihood = math.log(self.priorDistribution.getProbability(classLabel))
        for i in range(instance.attributeSize()):
            xi = instance.getAttribute(i).getValue()
            mi = self.__classMeans[classLabel].getValue(i)
            si = self.__classDeviations[classLabel].getValue(i)
            if si != 0:
                loglikelihood += -0.5 * math.pow((xi - mi) / si, 2)
        return loglikelihood

    def __logLikelihoodDiscrete(self, classLabel: str, instance: Instance) -> float:
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
        loglikelihood = math.log(self.priorDistribution.getProbability(classLabel))
        attributeDistributions = self.__classAttributeDistributions.get(classLabel)
        for i in range(instance.attributeSize()):
            xi = instance.getAttribute(i).getValue()
            loglikelihood += math.log(attributeDistributions[i].getProbabilityLaplaceSmoothing(xi))
        return loglikelihood

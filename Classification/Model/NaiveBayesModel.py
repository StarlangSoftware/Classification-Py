from Math.DiscreteDistribution import DiscreteDistribution

from Classification.Instance.Instance import Instance
from Classification.Model.GaussianModel import GaussianModel
import math


class NaiveBayesModel(GaussianModel):

    __class_means: dict
    __class_deviations: dict
    __class_attribute_distributions: dict

    def __init__(self, priorDistribution: DiscreteDistribution):
        """
        A constructor that sets the priorDistribution.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            DiscreteDistribution input.
        """
        self.prior_distribution = priorDistribution

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

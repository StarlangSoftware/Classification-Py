from Math.DiscreteDistribution import DiscreteDistribution

from Classification.Instance.Instance import Instance
from Classification.Model.GaussianModel import GaussianModel
import math


class NaiveBayesModel(GaussianModel):

    """
    A constructor that sets the priorDistribution.

    PARAMETERS
    ----------
    priorDistribution : DiscreteDistribution
        DiscreteDistribution input.
    """
    def __init__(self, priorDistribution: DiscreteDistribution):
        self.priorDistribution = priorDistribution

    """
    A constructor that sets the classMeans and classDeviations.

    PARAMETERS
    ----------
    classMeans : dict       
        A dict of String and Vector.
    classDeviations : dict  
        A dict of String and Vector.
    """
    def initForContinuous(self, classMeans: dict, classDeviations: dict):
        self.classMeans = classMeans
        self.classDeviations = classDeviations
        self.classAttributeDistributions = None

    """
    A constructor that sets the priorDistribution and classAttributeDistributions.

    PARAMETERS
    ----------
    classAttributeDistributions : dict
        A dict of String and list of DiscreteDistributions.
    """
    def initForDiscrete(self, classAttributeDistributions: dict):
        self.classAttributeDistributions = classAttributeDistributions

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
    def calculateMetric(self, instance: Instance, Ci: str) -> float:
        if self.classAttributeDistributions is None:
            return self.logLikelihoodContinuous(Ci, instance)
        else:
            return self.logLikelihoodDiscrete(Ci, instance)

    """
    The logLikelihoodContinuous method takes an Instance and a class label as inputs. First it gets the logarithm
    of given class label's probability via prior distribution as logLikelihood. Then it loops times of given instance 
    attribute size, and accumulates the logLikelihood by calculating -0.5 * ((xi - mi) / si )** 2).

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
    def logLikelihoodContinuous(self, classLabel: str, instance: Instance) -> float:
        loglikelihood = math.log(self.priorDistribution.getProbability(classLabel))
        for i in range(instance.attributeSize()):
            xi = instance.getAttribute(i).getValue()
            mi = self.classMeans[classLabel].getValue(i)
            si = self.classDeviations[classLabel].getValue(i)
            loglikelihood += -0.5 * math.pow((xi - mi) / si, 2)
        return loglikelihood

    """
    The logLikelihoodDiscrete method takes an Instance and a class label as inputs. First it gets the logarithm
    of given class label's probability via prior distribution as logLikelihood and gets the class attribute distribution 
    of given class label. Then it loops times of given instance attribute size, and accumulates the logLikelihood by 
    calculating the logarithm of corresponding attribute distribution's smoothed probability by using laplace smoothing 
    on xi.

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
    def logLikelihoodDiscrete(self, classLabel: str, instance: Instance) -> float:
        loglikelihood = math.log(self.priorDistribution.getProbability(classLabel))
        attributeDistributions = self.classAttributeDistributions.get(classLabel)
        for i in range(instance.attributeSize()):
            xi = instance.getAttribute(i).getValue()
            loglikelihood += math.log(attributeDistributions.get(i).getProbabilityLaplaceSmoothing(xi))
        return loglikelihood

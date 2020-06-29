from Math.DiscreteDistribution import DiscreteDistribution

from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.NaiveBayesModel import NaiveBayesModel
from Classification.Parameter.Parameter import Parameter


class NaiveBayes(Classifier):

    def trainContinuousVersion(self, priorDistribution: DiscreteDistribution, classLists: Partition):
        """
        Training algorithm for Naive Bayes algorithm with a continuous data set.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            Probability distribution of classes P(C_i)
        classLists : Partition
            Instances are divided into K lists, where each list contains only instances from a single class
        """
        classMeans = {}
        classDeviations = {}
        for i in range(classLists.size()):
            classLabel = classLists.get(i).getClassLabel()
            averageVector = classLists.get(i).average().toVector()
            classMeans[classLabel] = averageVector
            standardDeviationVector = classLists.get(i).standardDeviation().toVector()
            classDeviations[classLabel] = standardDeviationVector
        self.model = NaiveBayesModel(priorDistribution)
        if isinstance(self.model, NaiveBayesModel):
            self.model.initForContinuous(classMeans, classDeviations)

    def trainDiscreteVersion(self, priorDistribution: DiscreteDistribution, classLists: Partition):
        """
        Training algorithm for Naive Bayes algorithm with a discrete data set.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            Probability distribution of classes P(C_i)
        classLists : Partition
            Instances are divided into K lists, where each list contains only instances from a single class
        """
        classAttributeDistributions = {}
        for i in range(classLists.size()):
            classAttributeDistributions[classLists.get(i).getClassLabel()] = \
                classLists.get(i).allAttributesDistribution()
        self.model = NaiveBayesModel(priorDistribution)
        if isinstance(self.model, NaiveBayesModel):
            self.model.initForDiscrete(classAttributeDistributions)

    def train(self, trainSet: InstanceList, parameters: Parameter):
        """
        Training algorithm for Naive Bayes algorithm. It basically calls trainContinuousVersion for continuous data
        sets, trainDiscreteVersion for discrete data sets.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm
        """
        priorDistribution = trainSet.classDistribution()
        classLists = Partition(trainSet)
        if isinstance(classLists.get(0).get(0).getAttribute(0), DiscreteAttribute):
            self.trainDiscreteVersion(priorDistribution, classLists)
        else:
            self.trainContinuousVersion(priorDistribution, classLists)

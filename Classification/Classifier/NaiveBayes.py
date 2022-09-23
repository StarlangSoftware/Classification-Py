from Math.DiscreteDistribution import DiscreteDistribution

from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.NaiveBayesModel import NaiveBayesModel
from Classification.Parameter.Parameter import Parameter


class NaiveBayes(Classifier):

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
        self.model = NaiveBayesModel(priorDistribution)
        if isinstance(self.model, NaiveBayesModel):
            self.model.initForContinuous(class_means, class_deviations)

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
        self.model = NaiveBayesModel(priorDistribution)
        if isinstance(self.model, NaiveBayesModel):
            self.model.initForDiscrete(class_attribute_distributions)

    def train(self,
              trainSet: InstanceList,
              parameters: Parameter):
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

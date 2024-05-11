from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.Model import Model
from Math.DiscreteDistribution import DiscreteDistribution


class DummyModel(Model):

    distribution: DiscreteDistribution

    def constructor1(self, trainSet: InstanceList):
        """
        Constructor which sets the distribution using the given InstanceList.

        PARAMETERS
        ----------
        trainSet : InstanceList
            InstanceList which is used to get the class distribution.
        """
        self.distribution = trainSet.classDistribution()

    def constructor2(self, fileName: str):
        """
        Loads a dummy model from an input model file.
        :param fileName: Model file name.
        """
        inputFile = open(fileName, mode='r', encoding='utf-8')
        self.distribution = Model.loadClassDistribution(inputFile)
        inputFile.close()

    def __init__(self, trainSet: object):
        if isinstance(trainSet, InstanceList):
            self.constructor1(trainSet)
        elif isinstance(trainSet, str):
            self.constructor2(trainSet)

    def predict(self, instance: Instance) -> str:
        """
        The predict method takes an Instance as an input and returns the entry of distribution which has the maximum
        value.

        PARAMETERS
        ----------
        instance : Instance
            Instance to make prediction.

        RETURNS
        -------
        str
            The entry of distribution which has the maximum value.
        """
        if isinstance(instance, CompositeInstance):
            possible_class_labels = instance.getPossibleClassLabels()
            return self.distribution.getMaxItemIncludeTheseOnly(possible_class_labels)
        else:
            return self.distribution.getMaxItem()

    def predictProbability(self, instance: Instance) -> dict:
        """
        Calculates the posterior probability distribution for the given instance according to dummy model.
        :param instance: Instance for which posterior probability distribution is calculated.
        :return: Posterior probability distribution for the given instance.
        """
        return self.distribution.getProbabilityDistribution()

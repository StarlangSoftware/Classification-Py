from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.Model import Model


class DummyModel(Model):

    def __init__(self, trainSet: InstanceList):
        """
        Constructor which sets the distribution using the given InstanceList.

        PARAMETERS
        ----------
        trainSet : InstanceList
            InstanceList which is used to get the class distribution.
        """
        self.distribution = trainSet.classDistribution()

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
            possibleClassLabels = instance.getPossibleClassLabels()
            return self.distribution.getMaxItemIncludeTheseOnly(possibleClassLabels)
        else:
            return self.distribution.getMaxItem()

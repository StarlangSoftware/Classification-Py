from Classification.DataSet.DataSet import DataSet
from Classification.Filter.FeatureFilter import FeatureFilter
from Classification.Instance.Instance import Instance


class LaryFilter(FeatureFilter):

    def __init__(self, dataSet: DataSet):
        """
        Constructor that sets the dataSet and all the attributes distributions.

        PARAMETERS
        ----------
        dataSet : DataSet
            DataSet that will bu used.
        """
        super().__init__(dataSet)
        self.attributeDistributions = dataSet.getInstanceList().allAttributesDistribution()

    def removeDiscreteAttributesFromInstance(self, instance: Instance, size: int):
        """
        The removeDiscreteAttributesFromInstance method takes an Instance as an input, and removes the discrete
        attributes from given instance.

        PARAMETERS
        ----------
        instance : Instance
            Instance to removes attributes from.
        size : int
            Size of the given instance.
        """
        k = 0
        for i in range(size):
            if len(self.attributeDistributions[i]) > 0:
                instance.removeAttribute(k)
            else:
                k = k + 1

    def removeDiscreteAttributesFromDataDefinition(self, size: int):
        """
        The removeDiscreteAttributesFromDataDefinition method removes the discrete attributes from dataDefinition.

        PARAMETERS
        ----------
        size : int
            Size of item that attributes will be removed.
        """
        dataDefinition = self.dataSet.getDataDefinition()
        k = 0
        for i in range(size):
            if len(self.attributeDistributions[i]) > 0:
                dataDefinition.removeAttribute(k)
            else:
                k = k + 1

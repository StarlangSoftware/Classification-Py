from Classification.Attribute.DiscreteIndexedAttribute import DiscreteIndexedAttribute
from Classification.DataSet.DataSet import DataSet
from Classification.Filter.LaryFilter import LaryFilter
from Classification.Instance.Instance import Instance


class DiscreteToIndexed(LaryFilter):

    def __init__(self, dataSet: DataSet):
        """
        Constructor for discrete to indexed filter.

        PARAMETERS
        ----------
        dataSet : DataSet
            The dataSet whose instances whose discrete attributes will be converted to indexed attributes
        """
        super().__init__(dataSet)

    def convertInstance(self, instance: Instance):
        """
        Converts discrete attributes of a single instance to indexed version.

        PARAMETERS
        ----------
        instance : Instance
            The instance to be converted.
        """
        size = instance.attributeSize()
        for i in range(size):
            if len(self.attributeDistributions[i]) > 0:
                index = self.attributeDistributions[i].getIndex(instance.getAttribute(i).__str__())
                instance.addAttribute(DiscreteIndexedAttribute(instance.getAttribute(i).__str__(), index,
                                                               len(self.attributeDistributions[i])))
        self.removeDiscreteAttributesFromInstance(instance, size)

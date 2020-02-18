from Classification.Attribute.AttributeType import AttributeType
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.DataSet.DataSet import DataSet
from Classification.Filter.LaryFilter import LaryFilter
from Classification.Instance.Instance import Instance


class DiscreteToContinuous(LaryFilter):

    def __init__(self, dataSet: DataSet):
        """
        Constructor for discrete to continuous filter.

        PARAMETERS
        ----------
        dataSet : DataSet
            The dataSet whose instances whose discrete attributes will be converted to continuous attributes using
            1-of-L encoding.
        """
        super().__init__(dataSet)

    def convertInstance(self, instance: Instance):
        """
        Converts discrete attributes of a single instance to continuous version using 1-of-L encoding. For example, if
        an attribute has values red, green, blue; this attribute will be converted to 3 continuous attributes where
        red will have the value 100, green will have the value 010, and blue will have the value 001.

        PARAMETERS
        ----------
        instance : Instance
            The instance to be converted.
        """
        size = instance.attributeSize()
        for i in range(size):
            if len(self.attributeDistributions[i]) > 0:
                index = self.attributeDistributions[i].getIndex(instance.getAttribute(i).__str__())
                for j in range(len(self.attributeDistributions[i])):
                    if j != index:
                        instance.addAttribute(ContinuousAttribute(0))
                    else:
                        instance.addAttribute(ContinuousAttribute(1))
        self.removeDiscreteAttributesFromInstance(instance, size)

    def convertDataDefinition(self):
        """
        Converts the data definition with discrete attributes, to data definition with continuous attributes. Basically,
        for each discrete attribute with L possible values, L more continuous attributes will be added.
        """
        dataDefinition = self.dataSet.getDataDefinition()
        size = dataDefinition.attributeCount()
        for i in range(size):
            if len(self.attributeDistributions[i]) > 0:
                for j in range(len(self.attributeDistributions[i])):
                    dataDefinition.addAttribute(AttributeType.CONTINUOUS)
        self.removeDiscreteAttributesFromDataDefinition(size)

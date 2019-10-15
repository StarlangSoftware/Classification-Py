from Classification.Attribute.AttributeType import AttributeType
from Classification.Attribute.BinaryAttribute import BinaryAttribute
from Classification.DataSet.DataSet import DataSet
from Classification.Filter.LaryFilter import LaryFilter
from Classification.Instance.Instance import Instance


class LaryToBinary(LaryFilter):

    """
    Constructor for L-ary discrete to binary discrete filter.

    PARAMETERS
    ----------
    dataSet : DataSet
        The instances whose L-ary discrete attributes will be converted to binary discrete attributes.
    """
    def __init__(self, dataSet: DataSet):
        super().__init__(dataSet)

    """
    Converts discrete attributes of a single instance to binary discrete version using 1-of-L encoding. For example, if
    an attribute has values red, green, blue; this attribute will be converted to 3 binary attributes where
    red will have the value true false false, green will have the value false true false, and blue will have the value 
    false false true.

    PARAMETERS
    ----------
    instance : Instance
        The instance to be converted.
    """
    def convertInstance(self, instance: Instance):
        size = instance.attributeSize()
        for i in range(size):
            if len(self.attributeDistributions[i]) > 0:
                index = self.attributeDistributions[i].getIndex(instance.getAttribute(i).__str__())
                for j in range(len(self.attributeDistributions[i])):
                    if j != index:
                        instance.addAttribute(BinaryAttribute(False))
                    else:
                        instance.addAttribute(BinaryAttribute(True))
        self.removeDiscreteAttributesFromInstance(instance, size)

    """
    Converts the data definition with L-ary discrete attributes, to data definition with binary discrete attributes.
    """
    def convertDataDefinition(self):
        dataDefinition = self.dataSet.getDataDefinition()
        size = dataDefinition.attributeCount()
        for i in range(size):
            if len(self.attributeDistributions[i]):
                for j in range(len(self.attributeDistributions[i])):
                    dataDefinition.addAttribute(AttributeType.BINARY)
        self.removeDiscreteAttributesFromDataDefinition(size)

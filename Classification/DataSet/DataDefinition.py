from __future__ import annotations
from Classification.Attribute.AttributeType import AttributeType
from Classification.FeatureSelection.FeatureSubSet import FeatureSubSet


class DataDefinition(object):

    __attributeTypes: list

    """
    Constructor for creating a new DataDefinition with given attribute types.

    PARAMETERS
    ----------
    attributeTypes : list
        Attribute types of the data definition.
    """
    def __init__(self, attributeTypes=None):
        if attributeTypes is None:
            attributeTypes = []
        self.__attributeTypes = attributeTypes

    """
    Returns the number of attribute types.

    RETURNS
    -------
    int
        Number of attribute types.
    """
    def attributeCount(self) -> int:
        return len(self.__attributeTypes)

    """
    Counts the occurrences of binary and discrete type attributes.

    RETURNS
    -------
    int
        Count of binary and discrete type attributes.
    """
    def discreteAttributeCount(self) -> int:
        count = 0
        for attributeType in self.__attributeTypes:
            if attributeType is AttributeType.DISCRETE or attributeType is AttributeType.BINARY:
                count = count + 1
        return count

    """
    Counts the occurrences of binary and continuous type attributes.

    RETURNS
    -------
    int
        Count of of binary and continuous type attributes.
    """
    def continuousAttributeCount(self) -> int:
        count = 0
        for attributeType in self.__attributeTypes:
            if attributeType is AttributeType.CONTINUOUS:
                count = count + 1
        return count

    """
    Returns the attribute type of the corresponding item at given index.

    PARAMETERS
    ----------
    index : int 
        Index of the attribute type.
        
    RETURNS
    -------
    AttributeType
        Attribute type of the corresponding item at given index.
    """
    def getAttributeType(self, index: int) -> AttributeType:
        return self.__attributeTypes[index]

    """
    Adds an attribute type to the list of attribute types.

    PARAMETERS
    ----------
    attributeType : AttributeType
        Attribute type to add to the list of attribute types.
    """
    def addAttribute(self, attributeType: AttributeType):
        self.__attributeTypes.append(attributeType)

    """
    Removes the attribute type at given index from the list of attributes.

    PARAMETERS
    ----------
    index : int
        Index to remove attribute type from list.
    """
    def removeAttribute(self, index: int):
        self.__attributeTypes.pop(index)

    """
    Clears all the attribute types from list.
    """
    def removeAllAtrributes(self):
        self.__attributeTypes.clear()

    """
    Generates new subset of attribute types by using given feature subset.

    PARAMETERS
    ----------
    featureSubSet : FeatureSubSet
        FeatureSubSet input.
        
    RETURNS
    -------
    DataDefinition
        DataDefinition with new subset of attribute types.
    """
    def getSubSetOfFeatures(self, featureSubSet: FeatureSubSet) -> DataDefinition:
        newAttributeTypes = []
        for i in range(featureSubSet.size()):
            newAttributeTypes.append(self.__attributeTypes[featureSubSet.get(i)])
        return DataDefinition(newAttributeTypes)

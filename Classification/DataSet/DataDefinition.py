from __future__ import annotations
from Classification.Attribute.AttributeType import AttributeType
from Classification.FeatureSelection.FeatureSubSet import FeatureSubSet


class DataDefinition(object):

    __attributeTypes: list

    def __init__(self, attributeTypes=None):
        """
        Constructor for creating a new DataDefinition with given attribute types.

        PARAMETERS
        ----------
        attributeTypes : list
            Attribute types of the data definition.
        """
        if attributeTypes is None:
            attributeTypes = []
        self.__attributeTypes = attributeTypes

    def attributeCount(self) -> int:
        """
        Returns the number of attribute types.

        RETURNS
        -------
        int
            Number of attribute types.
        """
        return len(self.__attributeTypes)

    def discreteAttributeCount(self) -> int:
        """
        Counts the occurrences of binary and discrete type attributes.

        RETURNS
        -------
        int
            Count of binary and discrete type attributes.
        """
        count = 0
        for attributeType in self.__attributeTypes:
            if attributeType is AttributeType.DISCRETE or attributeType is AttributeType.BINARY:
                count = count + 1
        return count

    def continuousAttributeCount(self) -> int:
        """
        Counts the occurrences of binary and continuous type attributes.

        RETURNS
        -------
        int
            Count of of binary and continuous type attributes.
        """
        count = 0
        for attributeType in self.__attributeTypes:
            if attributeType is AttributeType.CONTINUOUS:
                count = count + 1
        return count

    def getAttributeType(self, index: int) -> AttributeType:
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
        return self.__attributeTypes[index]

    def addAttribute(self, attributeType: AttributeType):
        """
        Adds an attribute type to the list of attribute types.

        PARAMETERS
        ----------
        attributeType : AttributeType
            Attribute type to add to the list of attribute types.
        """
        self.__attributeTypes.append(attributeType)

    def removeAttribute(self, index: int):
        """
        Removes the attribute type at given index from the list of attributes.

        PARAMETERS
        ----------
        index : int
            Index to remove attribute type from list.
        """
        self.__attributeTypes.pop(index)

    def removeAllAtrributes(self):
        """
        Clears all the attribute types from list.
        """
        self.__attributeTypes.clear()

    def getSubSetOfFeatures(self, featureSubSet: FeatureSubSet) -> DataDefinition:
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
        newAttributeTypes = []
        for i in range(featureSubSet.size()):
            newAttributeTypes.append(self.__attributeTypes[featureSubSet.get(i)])
        return DataDefinition(newAttributeTypes)

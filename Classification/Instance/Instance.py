from __future__ import annotations
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.Attribute.Attribute import Attribute
from Math.Vector import Vector

from Classification.FeatureSelection.FeatureSubSet import FeatureSubSet


class Instance(object):

    __classLabel: str
    __attributes: list

    def __init__(self, classLabel: str, attributes=None):
        """
        Constructor for a single instance. Given the attributes and class label, it generates a new instance.

        PARAMETERS
        ----------
        classLabel : str
            Class label of the instance.
        attributes : list
            Attributes of the instance.
        """
        if attributes is None:
            attributes = []
        self.__classLabel = classLabel
        self.__attributes = attributes

    def __lt__(self, other):
        return self.__classLabel < other.classLabel

    def __gt__(self, other):
        return self.__classLabel > other.classLabel

    def __eq__(self, other):
        return self.__classLabel == other.classLabel

    def addDiscreteAttribute(self, value: str):
        """
        Adds a discrete attribute with the given String value.

        PARAMETERS
        ----------
        value : str
            Value of the discrete attribute.
        """
        self.__attributes.append(DiscreteAttribute(value))

    def addContinuousAttribute(self, value: float):
        """
        Adds a continuous attribute with the given float value.

        PARAMETERS
        ----------
        value : float
            Value of the continuous attribute.
        """
        self.__attributes.append(ContinuousAttribute(value))

    def addAttribute(self, attribute: Attribute):
        """
        Adds a new attribute.

        PARAMETERS
        ----------
        attribute : Attribute
            Attribute to be added.
        """
        self.__attributes.append(attribute)

    def addVectorAttribute(self, vector: Vector):
        """
        Adds a Vector of continuous attributes.

        PARAMETERS
        ----------
        vector : Vector
            Vector that has the continuous attributes.
        """
        for i in range(vector.size()):
            self.__attributes.append(ContinuousAttribute(vector.getValue(i)))

    def removeAttribute(self, index: int):
        """
        Removes attribute with the given index from the attributes list.

        PARAMETERS
        ----------
        index : int
            Index of the attribute to be removed.
        """
        self.__attributes.pop(index)

    def removeAllAttributes(self):
        """
        Removes all the attributes from the attributes list.
        """
        self.__attributes.clear()

    def getAttribute(self, index: int) -> Attribute:
        """
        Accessor for a single attribute.

        PARAMETERS
        ----------
        index : int
            Index of the attribute to be accessed.

        RETURNS
        -------
        Attribute
            Attribute with index 'index'.
        """
        return self.__attributes[index]

    def attributeSize(self) -> int:
        """
        Returns the number of attributes in the attributes list.

        RETURNS
        -------
        int
            Number of attributes in the attributes list.
        """
        return len(self.__attributes)

    def continuousAttributeSize(self) -> int:
        """
        Returns the number of continuous and discrete indexed attributes in the attributes list.

        RETURNS
        -------
        int
            Number of continuous and discrete indexed attributes in the attributes list.
        """
        size = 0
        for attribute in self.__attributes:
            size += attribute.continuousAttributeSize()
        return size

    def continuousAttributes(self) -> list:
        """
        The continuousAttributes method creates a new list result and it adds the continuous attributes of the
        attributes list and also it adds 1 for the discrete indexed attributes.

        RETURNS
        -------
        list
            result list that has continuous and discrete indexed attributes.
        """
        result = []
        for attribute in self.__attributes:
            result.extend(attribute.continuousAttributes())
        return result

    def getClassLabel(self) -> str:
        """
        Accessor for the class label.

        RETURNS
        -------
        str
            Class label of the instance.
        """
        return self.__classLabel

    def __str__(self) -> str:
        """
        Converts instance to a String.

        RETURNS
        -------
        str
            A string of attributes separated with comma character.
        """
        result = ""
        for attribute in self.__attributes:
            result = result + attribute.__str__() + ","
        result = result + self.__classLabel
        return result

    def getSubSetOfFeatures(self, featureSubSet: FeatureSubSet) -> Instance:
        """
        The getSubSetOfFeatures method takes a FeatureSubSet as an input. First it creates a result Instance
        with the class label, and adds the attributes of the given featureSubSet to it.

        PARAMETERS
        ----------
        featureSubSet : FeatureSubSet
            FeatureSubSet an list of indices.

        RETURNS
        -------
        Instance
            result Instance.
        """
        result = Instance(self.__classLabel)
        for i in range(featureSubSet.size()):
            result.addAttribute(self.__attributes[featureSubSet.get(i)])
        return result

    def toVector(self) -> Vector:
        """
        The toVector method returns a Vector of continuous attributes and discrete indexed attributes.

        RETURNS
        -------
        Vector
            Vector of continuous attributes and discrete indexed attributes.
        """
        values = []
        for attribute in self.__attributes:
            values.extend(attribute.continuousAttributes())
        return Vector(values)

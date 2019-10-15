from __future__ import annotations
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.Attribute.Attribute import Attribute
from Math.Vector import Vector

from Classification.FeatureSelection.FeatureSubSet import FeatureSubSet


class Instance(object):

    """
    Constructor for a single instance. Given the attributes and class label, it generates a new instance.

    PARAMETERS
    ----------
    classLabel : str
        Class label of the instance.
    attributes : list
        Attributes of the instance.
    """
    def __init__(self, classLabel: str, attributes=None):
        if attributes is None:
            attributes = []
        self.classLabel = classLabel
        self.attributes = attributes

    def __lt__(self, other):
        return self.classLabel < other.classLabel

    def __gt__(self, other):
        return self.classLabel > other.classLabel

    def __eq__(self, other):
        return self.classLabel == other.classLabel

    """
    Adds a discrete attribute with the given String value.

    PARAMETERS
    ----------
    value : str
        Value of the discrete attribute.
    """
    def addDiscreteAttribute(self, value: str):
        self.attributes.append(DiscreteAttribute(value))

    """
    Adds a continuous attribute with the given float value.

    PARAMETERS
    ----------
    value : float 
        Value of the continuous attribute.
    """
    def addContinuousAttribute(self, value: float):
        self.attributes.append(ContinuousAttribute(value))

    """
    Adds a new attribute.

    PARAMETERS
    ----------
    attribute : Attribute
        Attribute to be added.
    """
    def addAttribute(self, attribute: Attribute):
        self.attributes.append(attribute)

    """
    Adds a Vector of continuous attributes.

    PARAMETERS
    ----------
    vector : Vector
        Vector that has the continuous attributes.
    """
    def addVectorAttribute(self, vector: Vector):
        for i in range(vector.size()):
            self.attributes.append(ContinuousAttribute(vector.getValue(i)))

    """
    Removes attribute with the given index from the attributes list.

    PARAMETERS
    ----------
    index : int
        Index of the attribute to be removed.
    """
    def removeAttribute(self, index: int):
        self.attributes.pop(index)

    """
    Removes all the attributes from the attributes list.
    """
    def removeAllAttributes(self):
        self.attributes.clear()

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
    def getAttribute(self, index: int) -> Attribute:
        return self.attributes[index]

    """
    Returns the number of attributes in the attributes list.

    RETURNS
    -------
    int
        Number of attributes in the attributes list.
    """
    def attributeSize(self) -> int:
        return len(self.attributes)

    """
    Returns the number of continuous and discrete indexed attributes in the attributes list.

    RETURNS
    -------
    int
        Number of continuous and discrete indexed attributes in the attributes list.
    """
    def continuousAttributeSize(self) -> int:
        size = 0
        for attribute in self.attributes:
            size += attribute.continuousAttributeSize()
        return size

    """
    The continuousAttributes method creates a new list result and it adds the continuous attributes of the
    attributes list and also it adds 1 for the discrete indexed attributes.

    RETURNS
    -------
    list
        result list that has continuous and discrete indexed attributes.
    """
    def continuousAttributes(self) -> list:
        result = []
        for attribute in self.attributes:
            result.append(attribute.continuousAttributes())
        return result

    """
    Accessor for the class label.

    RETURNS
    -------
    str
        Class label of the instance.
    """
    def getClassLabel(self) -> str:
        return self.classLabel

    """
    Converts instance to a String.

    RETURNS
    -------
    str
        A string of attributes separated with comma character.
    """
    def __str__(self) -> str:
        result = ""
        for attribute in self.attributes:
            result = result + attribute.__str__() + ","
        result = result + self.classLabel
        return result

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
    def getSubSetOfFeatures(self, featureSubSet: FeatureSubSet) -> Instance:
        result = Instance(self.classLabel)
        for i in range(featureSubSet.size()):
            result.addAttribute(self.attributes[featureSubSet.get(i)])
        return result

    """
    The toVector method returns a Vector of continuous attributes and discrete indexed attributes.

    RETURNS
    -------
    Vector
    Vector of continuous attributes and discrete indexed attributes.
    """
    def toVector(self) -> Vector:
        values = []
        for attribute in self.attributes:
            values.append(attribute.continuousAttributes())
        return Vector(values)

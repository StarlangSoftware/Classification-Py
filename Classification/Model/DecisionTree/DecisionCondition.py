from Classification.Attribute.Attribute import Attribute
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.DiscreteIndexedAttribute import DiscreteIndexedAttribute
from Classification.Instance.Instance import Instance


class DecisionCondition(object):

    """
    A constructor that sets attributeIndex and Attribute value. It also assigns equal sign to the comparison character.

    PARAMETERS
    ----------
    attributeIndex : int
        Integer number that shows attribute index.
    value : Attribute
        The value of the Attribute.
    """
    def __init__(self, attributeIndex: int, value: Attribute, comparison="="):
        self.attributeIndex = attributeIndex
        self.comparison = comparison
        self.value = value

    """
    The satisfy method takes an Instance as an input.

    If defined Attribute value is a DiscreteIndexedAttribute it compares the index of Attribute of instance at the
    attributeIndex and the index of Attribute value and returns the result.

    If defined Attribute value is a DiscreteAttribute it compares the value of Attribute of instance at the
    attributeIndex and the value of Attribute value and returns the result.

    If defined Attribute value is a ContinuousAttribute it compares the value of Attribute of instance at the
    attributeIndex and the value of Attribute value and returns the result according to the comparison character 
    whether it is less than or greater than signs.

    PARAMETERS
    ----------
    instance : Instance
        Instance to compare.
        
    RETURNS
    -------
    bool
        True if gicen instance satisfies the conditions.
    """
    def satisfy(self, instance: Instance):
        if isinstance(self.value, DiscreteIndexedAttribute):
            if self.value.getIndex() != -1:
                return instance.getAttribute(self.attributeIndex).getIndex() == self.value.getIndex()
            else:
                return True
        elif isinstance(self.value, DiscreteAttribute):
            return instance.getAttribute(self.attributeIndex).getValue() == self.value.getValue()
        elif isinstance(self.value, ContinuousAttribute):
            if self.comparison == "<":
                return instance.getAttribute(self.attributeIndex).getValue() <= self.value.getValue()
            else:
                return instance.getAttribute(self.attributeIndex).getValue() > self.value.getValue()
        return False

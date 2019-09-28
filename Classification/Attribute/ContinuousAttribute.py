from Classification.Attribute.Attribute import Attribute


class ContinuousAttribute(Attribute):

    """
    Constructor for a continuous attribute.

    PARAMETERS
    ----------
    value : str
        Value of the attribute.
    """
    def __init__(self, value: float):
        self.value = value

    """
    Accessor method for value.

    RETURNS
    -------
    float
        value
    """
    def getValue(self) -> float:
        return self.value

    """
    Mutator method for value

    PARAMETERS
    ----------
    value : float
        New value of value.
    """
    def setValue(self, value: float):
        self.value = value

    """
    Converts value to {@link String}.

    RETURNS
    -------
    str
        String representation of value.
    """
    def __str__(self) -> str:
        return self.value

    def continuousAttributeSize(self) -> int:
        return 1

    def continuousAttributes(self) -> list:
        result = []
        result.append(self.value)
        return result
from Classification.Attribute.Attribute import Attribute


class DiscreteAttribute(Attribute):

    """
    Constructor for a discrete attribute.

    PARAMETERS
    ----------
    value : str
        Value of the attribute.
    """
    def __init__(self, value: str):
        self.value = value

    """
    Accessor method for value.

    RETURNS
    -------
    str
        value
    """
    def getValue(self) -> str:
        return self.value

    """
    Converts value to String.

    RETURNS
    -------
    str
        String representation of value.
    """
    def __str__(self) -> str:
        if self.value == ",":
            return "comma"
        return self.value

    def continuousAttributeSize(self) -> int:
        return 0

    def continuousAttributes(self) -> list:
        return []
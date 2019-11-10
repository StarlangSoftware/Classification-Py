from Classification.Attribute.Attribute import Attribute


class ContinuousAttribute(Attribute):

    __value: float

    """
    Constructor for a continuous attribute.

    PARAMETERS
    ----------
    value : str
        Value of the attribute.
    """
    def __init__(self, value: float):
        self.__value = value

    """
    Accessor method for value.

    RETURNS
    -------
    float
        value
    """
    def getValue(self) -> float:
        return self.__value

    """
    Mutator method for value

    PARAMETERS
    ----------
    value : float
        New value of value.
    """
    def setValue(self, value: float):
        self.__value = value

    """
    Converts value to {@link String}.

    RETURNS
    -------
    str
        String representation of value.
    """
    def __str__(self) -> str:
        return self.__value.__str__()

    def continuousAttributeSize(self) -> int:
        return 1

    def continuousAttributes(self) -> list:
        result = []
        result.append(self.__value)
        return result
from Classification.Attribute.Attribute import Attribute


class ContinuousAttribute(Attribute):

    __value: float

    def __init__(self, value: float):
        """
        Constructor for a continuous attribute.

        PARAMETERS
        ----------
        value : str
            Value of the attribute.
        """
        self.__value = value

    def getValue(self) -> float:
        """
        Accessor method for value.

        RETURNS
        -------
        float
            value
        """
        return self.__value

    def setValue(self, value: float):
        """
        Mutator method for value

        PARAMETERS
        ----------
        value : float
            New value of value.
        """
        self.__value = value

    def __str__(self) -> str:
        """
        Converts value to {@link String}.

        RETURNS
        -------
        str
            String representation of value.
        """
        return self.__value.__str__()

    def continuousAttributeSize(self) -> int:
        return 1

    def continuousAttributes(self) -> list:
        result = [self.__value]
        return result

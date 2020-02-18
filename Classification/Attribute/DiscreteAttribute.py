from Classification.Attribute.Attribute import Attribute


class DiscreteAttribute(Attribute):

    __value: str

    def __init__(self, value: str):
        """
        Constructor for a discrete attribute.

        PARAMETERS
        ----------
        value : str
            Value of the attribute.
        """
        self.__value = value

    def getValue(self) -> str:
        """
        Accessor method for value.

        RETURNS
        -------
        str
            value
        """
        return self.__value

    def __str__(self) -> str:
        """
        Converts value to String.

        RETURNS
        -------
        str
            String representation of value.
        """
        if self.__value == ",":
            return "comma"
        return self.__value

    def continuousAttributeSize(self) -> int:
        return 0

    def continuousAttributes(self) -> list:
        return []

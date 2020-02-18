from Classification.Attribute.DiscreteAttribute import DiscreteAttribute


class DiscreteIndexedAttribute(DiscreteAttribute):

    __index: int
    __maxIndex: int

    def __init__(self, value: str, index: int, maxIndex: int):
        """
        Constructor for a discrete attribute.

        PARAMETERS
        ----------
        value : str
            Value of the attribute.
        index : int
            Index of the attribute.
        maxIndex : int
            Maximum index of the attribute.
        """
        super().__init__(value)
        self.__index = index
        self.__maxIndex = maxIndex

    def getIndex(self) -> int:
        """
        Accessor method for index.

        RETURNS
        -------
        int
            index.
        """
        return self.__index

    def getMaxIndex(self) -> int:
        """
        Accessor method for maxIndex.

        RETURNS
        -------
        int
            maxIndex.
        """
        return self.__maxIndex

    def continuousAttributeSize(self) -> int:
        return self.__maxIndex

    def continuousAttributes(self) -> list:
        result = []
        for i in range(self.__maxIndex):
            if i != self.__index:
                result.append(0.0)
            else:
                result.append(1.0)
        return result

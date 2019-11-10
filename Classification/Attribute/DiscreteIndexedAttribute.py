from Classification.Attribute.DiscreteAttribute import DiscreteAttribute


class DiscreteIndexedAttribute(DiscreteAttribute):

    __index: int
    __maxIndex: int

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
    def __init__(self, value : str, index: int, maxIndex: int):
        super().__init__(value)
        self.__index = index
        self.__maxIndex = maxIndex

    """
    Accessor method for index.

    RETURNS
    -------
    int
        index.
    """
    def getIndex(self) -> int:
        return self.__index

    """
    Accessor method for maxIndex.
    
    RETURNS
    -------
    int
        maxIndex.
    """
    def getMaxIndex(self) -> int:
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

class FeatureSubSet(object):

    __indexList: list

    """
    A constructor that sets the indexList {@link ArrayList}.

    PARAMETERS
    ----------
    indexList : list
        An ArrayList consists of integer indices.
    """
    def __init__(self, indexList=None):
        if indexList is None:
            indexList = []
        self.__indexList = indexList

    """
    A constructor that takes number of features as input and initializes indexList with these numbers.

    PARAMETERS
    ----------
    numberOfFeatures : int
        Indicates the indices of indexList.
    """
    def initWithNumberOfFeatures(self, numberOfFeatures: int):
        self.__indexList = [i for i in range(numberOfFeatures)]

    """
    The size method returns the size of the indexList.

    RETURNS
    -------
    int
        The size of the indexList.
    """
    def size(self) -> int:
        return len(self.__indexList)

    """
    The get method returns the item of indexList at given index.

    PARAMETERS
    ----------
    index : int
        Index of the indexList to be accessed.
        
    RETURNS
    -------
    int
        The item of indexList at given index.
    """
    def get(self, index: int) -> int:
        return self.__indexList[index]

    """
    The contains method returns True, if indexList contains given input number and False otherwise.

    PARAMETERS
    ----------
    featureNo : int 
        Feature number that will be checked.
        
    RETURNS
    -------
    bool        
        True, if indexList contains given input number.
    """
    def contains(self, featureNo: int) -> bool:
        return featureNo in self.__indexList

    """
    The add method adds given Integer to the indexList.

    PARAMETERS
    ----------
    featureNo : int
        Integer that will be added to indexList.
    """
    def add(self, featureNo: int):
        self.__indexList.append(featureNo)

    """
    The remove method removes the item of indexList at the given index.

    PARAMETERS
    ----------
    index : int
        Index of the item that will be removed.
    """
    def remove(self, index: int):
        self.__indexList.remove(index)

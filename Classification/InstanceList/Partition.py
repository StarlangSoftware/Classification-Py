from Classification.InstanceList.InstanceList import InstanceList


class Partition(object):

    __multilist: list

    """
    Constructor for generating a partition.
    """
    def __init__(self):
        self.__multilist = []

    """
    Adds given instance list to the list of instance lists.

    PARAMETERS
    ----------
    list : InstanceList
        Instance list to add.
    """
    def add(self, list: InstanceList):
        self.__multilist.append(list)

    """
    Returns the size of the list of instance lists.

    RETURNS
    -------
    int
        The size of the list of instance lists.
    """
    def size(self) -> int:
        return len(self.__multilist)

    """
    Returns the corresponding instance list at given index of list of instance lists.

    PARAMETERS
    ----------
    index : int
        Index of the instance list.
        
    RETURNS
    -------
    InstanceList
        Instance list at given index of list of instance lists.
    """
    def get(self, index: int) -> InstanceList:
        return self.__multilist[index]

    """
    Returns the instances of the items at the list of instance lists.

    RETURNS
    -------
    list
        Instances of the items at the list of instance lists.
    """
    def getLists(self) -> list:
        result = []
        for instanceList in self.__multilist:
            result.append(instanceList.getInstances())
        return result
from Classification.InstanceList.InstanceList import InstanceList


class Partition(object):

    __multilist: list

    def __init__(self):
        """
        Constructor for generating a partition.
        """
        self.__multilist = []

    def add(self, _list: InstanceList):
        """
        Adds given instance list to the list of instance lists.

        PARAMETERS
        ----------
        _list : InstanceList
            Instance list to add.
        """
        self.__multilist.append(_list)

    def size(self) -> int:
        """
        Returns the size of the list of instance lists.

        RETURNS
        -------
        int
            The size of the list of instance lists.
        """
        return len(self.__multilist)

    def get(self, index: int) -> InstanceList:
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
        return self.__multilist[index]

    def getLists(self) -> list:
        """
        Returns the instances of the items at the list of instance lists.

        RETURNS
        -------
        list
            Instances of the items at the list of instance lists.
        """
        result = []
        for instanceList in self.__multilist:
            result.append(instanceList.getInstances())
        return result

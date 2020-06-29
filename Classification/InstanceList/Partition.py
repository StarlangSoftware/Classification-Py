import random

from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.InstanceListOfSameClass import InstanceListOfSameClass


class Partition(object):

    __multilist: list

    def __init__(self, instanceList: InstanceList = None, ratio=None, seed=None, stratified: bool = None):
        """
        Divides the instances in the instance list into partitions so that all instances of a class are grouped in a
        single partition.
        PARAMETERS
        ----------
        ratio
            Ratio of the stratified partition. Ratio is between 0 and 1. If the ratio is 0.2, then 20 percent of the
            instances are put in the first group, 80 percent of the instances are put in the second group.
        seed
            seed is used as a random number.
        """
        self.__multilist = []
        if instanceList is not None:
            if ratio is None:
                classLabels = instanceList.getDistinctClassLabels()
                for classLabel in classLabels:
                    self.add(InstanceListOfSameClass(classLabel))
                for instance in instanceList.getInstances():
                    self.get(classLabels.index(instance.getClassLabel())).add(instance)
            else:
                if isinstance(ratio, float):
                    self.add(InstanceList())
                    self.add(InstanceList())
                    if stratified:
                        distribution = instanceList.classDistribution()
                        counts = [0] * len(distribution)
                        randomArray = [i for i in range(instanceList.size())]
                        random.seed(seed)
                        random.shuffle(randomArray)
                        for i in range(instanceList.size()):
                            instance = instanceList.get(randomArray[i])
                            classIndex = distribution.getIndex(instance.getClassLabel())
                            if counts[classIndex] < instanceList.size() * ratio * \
                                    distribution.getProbability(instance.getClassLabel()):
                                self.get(0).add(instance)
                            else:
                                self.get(1).add(instance)
                            counts[classIndex] = counts[classIndex] + 1
                    else:
                        instanceList.shuffle(seed)
                        for i in range(self.size()):
                            instance = instanceList.get(i)
                            if i < instanceList.size() * ratio:
                                self.get(0).add(instance)
                            else:
                                self.get(1).add(instance)
                elif isinstance(ratio, int):
                    attributeIndex = ratio
                    if seed is None:
                        valueList = instanceList.getAttributeValueList(attributeIndex)
                        for _ in valueList:
                            self.add(InstanceList())
                        for instance in instanceList.getInstances():
                            self.get(valueList.index(instance.getAttribute(attributeIndex).getValue())).add(instance)
                    elif isinstance(seed, int):
                        attributeValue = seed
                        self.add(InstanceList())
                        self.add(InstanceList())
                        for instance in instanceList.getInstances():
                            if instance.getAttribute(attributeIndex).getIndex() == attributeValue:
                                self.get(0).add(instance)
                            else:
                                self.get(1).add(instance)
                    elif isinstance(seed, float):
                        splitValue = seed
                        self.add(InstanceList())
                        self.add(InstanceList())
                        for instance in instanceList.getInstances():
                            if instance.getAttribute(attributeIndex).getValue() < splitValue:
                                self.get(0).add(instance)
                            else:
                                self.get(1).add(instance)

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

from __future__ import annotations
import random
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.Instance.Instance import Instance
from Classification.Attribute.AttributeType import AttributeType
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.BinaryAttribute import BinaryAttribute
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Sampling.Bootstrap import Bootstrap


class InstanceList(object):

    """
    Empty constructor for an instance list. Initializes the instance list with the given instance list.

    PARAMETERS
    ----------
    list : list
        New list for the list variable.
    """
    def __init__(self, list=None):
        if list is None:
            list = []
        self.list = list

    """
    Constructor for an instance list with a given data definition, data file and a separator character. Each instance
    must be stored in a separate line separated with the character separator. The last item must be the class label.
    The function reads the file line by line and for each line; depending on the data definition, that is, type of
    the attributes, adds discrete and continuous attributes to a new instance. For example, given the data set file

    red;1;0.4;true
    green;-1;0.8;true
    blue;3;1.3;false

    where the first attribute is a discrete attribute, second and third attributes are continuous attributes, the
    fourth item is the class label.

    PARAMETERS
    ----------
    definition : DataDefinition
        Data definition of the data set.
    separator : str 
        Separator character which separates the attribute values in the data file.
    fileName : str  
        Name of the data set file.
    """
    def initWithFile(self, definition: DataDefinition, separator: str, fileName: str):
        self.list = []
        file = open(fileName, 'r')
        lines = file.readlines()
        for line in lines:
            attributeList = line.split(separator)
            if len(attributeList) == definition.attributeCount():
                current = Instance(attributeList[len(attributeList) - 1])
                for i in range(len(attributeList) - 1):
                    if definition.getAttributeType(i) is AttributeType.DISCRETE:
                        current.addAttribute(DiscreteAttribute(attributeList[i]))
                    elif definition.getAttributeType(i) is AttributeType.BINARY:
                        current.addAttribute(BinaryAttribute(attributeList[i] in ["True", "true", "Yes", "yes", "y", "Y"]))
                    elif definition.getAttributeType(i) is AttributeType.CONTINUOUS:
                        current.addAttribute(ContinuousAttribute(float(attributeList[i])))
                self.list.append(current)

    """
    Adds instance to the instance list.

    PARAMETERS
    ----------
    instance : Instance
        Instance to be added.
    """
    def add(self, instance: Instance):
        self.list.append(instance)

    """
    Adds a list of instances to the current instance list.

    PARAMETERS
    ----------
    instanceList : list
        List of instances to be added.
    """
    def addAll(self, instanceList: list):
        self.list.extend(instanceList)

    """
    Returns size of the instance list.

    RETURNS
    -------
    int
        Size of the instance list.
    """
    def size(self) -> int:
        return len(self.list)

    """
    Accessor for a single instance with the given index.

    PARAMETERS
    ----------
    index : int
        Index of the instance.
        
    RETURNS
    -------
    Instance
        Instance with index 'index'.
    """
    def get(self, index: int) -> Instance:
        return self.list[index]

    def makeComparator(self, attributeIndex : int):
        def compare(instanceA, instanceB):
            result1 = instanceA.getAttribute(attributeIndex)
            result2 = instanceB.getAttribute(attributeIndex)
            if result1 < result2:
                return -1
            elif result1 > result2:
                return 1
            else:
                return 0
        return compare

    """
    Sorts attribute list according to the attribute with index 'attributeIndex'.

    PARAMETERS
    ----------
    attributeIndex : int
        index of the attribute.
    """
    def sortWrtAttribute(self, attributeIndex: int):
        self.list.sort(key=self.makeComparator(attributeIndex))

    """
    Sorts attributes list.
    """
    def sort(self):
        self.list.sort()

    """
    Shuffles the instance list.
    
    PARAMETERS
    ----------
    seed : int
        Seed is used for random number generation.
    """
    def shuffle(self, seed: int):
        random.shuffle(self.list, seed)

    """
    Creates a bootstrap sample from the current instance list.

    PARAMETERS
    ----------
    seed : int
        To create a different bootstrap sample, we need a new seed for each sample.
        
    RETURNS
    -------
    Bootstrap
        Bootstrap sample.
    """
    def bootstrap(self, seed: int) -> Bootstrap:
        return Bootstrap(self.list, seed)

    """
    Extracts the class labels of each instance in the instance list and returns them in an array of {@link String}.

    RETURNS
    -------
    list
        An array list of class labels.
    """
    def getClassLabels(self) -> list:
        classLabels = []
        for instance in self.list:
            classLabels.append(instance.getClassLabel())
        return classLabels
from Classification.InstanceList.InstanceList import InstanceList
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.Attribute.AttributeType import AttributeType
from Classification.Instance.Instance import Instance
from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.BinaryAttribute import BinaryAttribute
from Classification.Attribute.DiscreteIndexedAttribute import DiscreteIndexedAttribute


class DataSet(object):

    """
    Constructor for generating a new DataSet with given DataDefinition.

    PARAMETERS
    ----------
    definition : DataDefinition
        Data definition of the data set.
    """
    def __init__(self, definition:None):
        self.definition = definition
        self.instances = InstanceList()

    """
    Constructor for generating a new DataSet from given File.

    PARAMETERS
    ----------
    fileName : str 
        File to generate DataSet from.
    """
    def initWithFile(self, fileName: str):
        self.instances = InstanceList()
        self.definition = DataDefinition()
        input = open(fileName, 'r')
        lines = input.readlines()
        i = 0
        for line in lines:
            attributes = line.split(",")
            if i == 0:
                for j in range(len(attributes) - 1):
                    try:
                        float(attributes[j])
                        self.definition.addAttribute(AttributeType.CONTINUOUS)
                    except:
                        self.definition.addAttribute(AttributeType.DISCRETE)
            else:
                if len(attributes) != self.definition.attributeCount() + 1:
                    continue
            if not ";" in attributes[len(attributes) - 1]:
                instance = Instance(attributes[len(attributes) - 1])
            else:
                labels = attributes[len(attributes) - 1].split(";")
                instance = CompositeInstance(labels[0], None, labels)
            for j in range(len(attributes) - 1):
                if self.definition.getAttributeType(j) is AttributeType.CONTINUOUS:
                    instance.addAttribute(ContinuousAttribute(float(attributes[j])))
                elif self.definition.getAttributeType(j) is AttributeType.DISCRETE:
                    instance.addAttribute(DiscreteAttribute(attributes[j]))
            if instance.attributeSize() == self.definition.attributeCount():
                self.instances.add(instance)
            i = i + 1

    """
    Constructor for generating a new DataSet with a DataDefinition, from a File by using a separator.

    PARAMETERS
    ----------
    definition : DataDefinition
        Data definition of the data set.
    separator : str 
        Separator character which separates the attribute values in the data file.
    fileName : str  
        Name of the data set file.
    """
    def initWithDefinitionAndFile(self, definition: DataDefinition, separator: str, fileName: str):
        self.definition = definition
        self.instances = InstanceList()
        self.instances.initWithDefinitionAndFile(definition, separator, fileName)

    """
    Checks the correctness of the attribute type, for instance, if the attribute of given instance is a Binary attribute,
    and the attribute type of the corresponding item of the data definition is also a Binary attribute, it then
    returns true, and false otherwise.

    PARAMETERS
    ----------
    instance : Instance
        Instance to checks the attribute type.
        
    RETURNS
    -------
    bool
        true if attribute types of given Instance and data definition matches.
    """
    def checkDefinition(self, instance: Instance) -> bool:
        for i in range(instance.attributeSize()):
            if isinstance(instance.getAttribute(i), BinaryAttribute):
                if self.definition.getAttributeType(i) is not AttributeType.BINARY:
                    return False
            elif isinstance(instance.getAttribute(i), DiscreteIndexedAttribute):
                if self.definition.getAttributeType(i) is not AttributeType.DISCRETE_INDEXED:
                    return False
            elif isinstance(instance.getAttribute(i), DiscreteAttribute):
                if self.definition.getAttributeType(i) is not AttributeType.DISCRETE:
                    return False
            elif isinstance(instance.getAttribute(i), ContinuousAttribute):
                if self.definition.getAttributeType(i) is not AttributeType.CONTINUOUS:
                    return False
        return True

    """
    Adds the attribute types according to given Instance. For instance, if the attribute type of given Instance
    is a Discrete type, it than adds a discrete attribute type to the list of attribute types.

    PARAMETERS
    ----------
    instance : Instance
        Instance input.
    """
    def setDefinition(self, instance: Instance):
        attributeTypes = []
        for i in range(instance.attributeSize()):
            if isinstance(instance.getAttribute(i), BinaryAttribute):
                attributeTypes.append(AttributeType.BINARY)
            elif isinstance(instance.getAttribute(i), DiscreteIndexedAttribute):
                attributeTypes.append(AttributeType.DISCRETE_INDEXED)
            elif isinstance(instance.getAttribute(i), DiscreteAttribute):
                attributeTypes.append(AttributeType.DISCRETE)
            elif isinstance(instance.getAttribute(i), ContinuousAttribute):
                attributeTypes.append(AttributeType.CONTINUOUS)
        self.definition = DataDefinition(attributeTypes)

    """
    Returns the size of the InstanceList.

    RETURNS
    -------
    int
        Size of the InstanceList.
    """
    def sampleSize(self) -> int:
        return self.instances.size()

    """
    Returns the size of the class label distribution of InstanceList.

    RETURNS
    -------
    int
        Size of the class label distribution of InstanceList.
    """
    def classCount(self) -> int:
        return len(self.instances.classDistribution())

    """
    Returns the number of attribute types at DataDefinition list.

    RETURNS
    -------
    int
        The number of attribute types at DataDefinition list.
    """
    def attributeCount(self) -> int:
        return self.definition.attributeCount()

    """
    Returns the number of discrete attribute types at DataDefinition list.

    RETURNS
    -------
    int
        The number of discrete attribute types at DataDefinition list.
    """
    def discreteAttributeCount(self) -> int:
        return self.definition.discreteAttributeCount()

    """
    Returns the number of continuous attribute types at DataDefinition list.

    RETURNS
    -------
    int
        The number of continuous attribute types at DataDefinition list.
    """
    def continuousAttributeCount(self) -> int:
        return self.definition.continuousAttributeCount()

    """
    Returns the accumulated String of class labels of the InstanceList.

    RETURNS
    -------
    str
        The accumulated String of class labels of the InstanceList.
    """
    def getClasses(self) -> str:
        classLabels = self.instances.getDistinctClassLabels()
        result = classLabels[0]
        for i in range(len(classLabels)):
            result = result + ";" + classLabels[i]
        return result

    """
    Returns the general information about the given data set such as the number of instances, distinct class labels,
    attributes, discrete and continuous attributes.

    PARAMETERS
    ----------
    dataSetName : str
        Data set name.
        
    RETURNS
    -------
    str
        General information about the given data set.
    """
    def info(self, dataSetName: str) -> str:
        result = "DATASET: " + dataSetName + "\n"
        result = result + "Number of instances: " + self.sampleSize().__str__() + "\n"
        result = result + "Number of distinct class labels: " + self.classCount().__str__() + "\n"
        result = result + "Number of attributes: " + self.attributeCount().__str__() + "\n"
        result = result + "Number of discrete attributes: " + self.discreteAttributeCount().__str__() + "\n"
        result = result + "Number of continuous attributes: " + self.continuousAttributeCount().__str__() + "\n"
        result = result + "Class labels: " + self.getClasses()
        return result

    """
    Adds a new instance to the InstanceList.

    PARAMETERS
    ----------
    current : Instance
        Instance to add.
    """
    def addInstance(self, current: Instance):
        if self.definition is None:
            self.setDefinition(current)
            self.instances.add(current)
        elif self.checkDefinition(current):
            self.instances.add(current)

    """
    Adds all the instances of given instance list to the InstanceList.

    PARAMETERS
    ----------
    instanceList : list
        InstanceList to add instances from.
    """
    def addInstanceList(self, instanceList : list):
        for instance in instanceList:
            self.addInstance(instance)

    """
    Returns the instances of InstanceList.

    RETURNS
    -------
    list
        The instances of InstanceList.
    """
    def getInstances(self) -> list:
        return self.instances.getInstances()

    """
    Returns instances of the items at the list of instance lists from the partitions.

    RETURNS
    -------
    list
        Instances of the items at the list of instance lists from the partitions.
    """
    def getClassInstances(self) -> list:
        return self.instances.divideIntoClasses().getLists()

    """
    Accessor for the InstanceList.

    RETURNS
    -------
    InstanceList
        The InstanceList.
    """
    def getInstanceList(self) -> InstanceList:
        return self.instances

    """
    Accessor for the data definition.

    RETURNS
    -------
    DataDefinition
        The data definition.
    """
    def getDataDefinition(self) -> DataDefinition:
        return self.definition

    """
    Print out the instances of InstanceList as a String.

    PARAMETERS
    ----------
    outFileName : str
        File name to write the output.
    """
    def writeToFile(self, outFileName: str):
        outfile = open(outFileName, "w")
        for i in range(self.instances.size()):
            outfile.write(self.instances.get(i).__str__() + "\n")
        outfile.close()
from __future__ import annotations
from Classification.FeatureSelection.FeatureSubSet import FeatureSubSet
from Classification.InstanceList.InstanceList import InstanceList
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.Attribute.AttributeType import AttributeType
from Classification.Instance.Instance import Instance
from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.BinaryAttribute import BinaryAttribute
from Classification.Attribute.DiscreteIndexedAttribute import DiscreteIndexedAttribute
from Classification.InstanceList.Partition import Partition


class DataSet(object):
    __instances: InstanceList
    __definition: DataDefinition

    def __init__(self, definition: DataDefinition = None, separator: str = None, fileName: str = None):
        """
        Constructor for generating a new DataSet with given DataDefinition.

        PARAMETERS
        ----------
        definition : DataDefinition
            Data definition of the data set.
        separator : str
            Separator character which separates the attribute values in the data file.
        fileName : str
            Name of the data set file.
        """
        self.__definition = definition
        if separator is None:
            self.__instances = InstanceList()
        else:
            self.__instances = InstanceList(definition, separator, fileName)

    def initWithFile(self, fileName: str):
        """
        Constructor for generating a new DataSet from given File.

        PARAMETERS
        ----------
        fileName : str
            File to generate DataSet from.
        """
        self.__instances = InstanceList()
        self.__definition = DataDefinition()
        inputFile = open(fileName, 'r', encoding='utf8')
        lines = inputFile.readlines()
        i = 0
        for line in lines:
            attributes = line.split(",")
            if i == 0:
                for j in range(len(attributes) - 1):
                    try:
                        float(attributes[j])
                        self.__definition.addAttribute(AttributeType.CONTINUOUS)
                    except:
                        self.__definition.addAttribute(AttributeType.DISCRETE)
            else:
                if len(attributes) != self.__definition.attributeCount() + 1:
                    continue
            if ";" not in attributes[len(attributes) - 1]:
                instance = Instance(attributes[len(attributes) - 1])
            else:
                labels = attributes[len(attributes) - 1].split(";")
                instance = CompositeInstance(labels[0], None, labels)
            for j in range(len(attributes) - 1):
                if self.__definition.getAttributeType(j) is AttributeType.CONTINUOUS:
                    instance.addAttribute(ContinuousAttribute(float(attributes[j])))
                elif self.__definition.getAttributeType(j) is AttributeType.DISCRETE:
                    instance.addAttribute(DiscreteAttribute(attributes[j]))
            if instance.attributeSize() == self.__definition.attributeCount():
                self.__instances.add(instance)
            i = i + 1

    def __checkDefinition(self, instance: Instance) -> bool:
        """
        Checks the correctness of the attribute type, for instance, if the attribute of given instance is a Binary
        attribute, and the attribute type of the corresponding item of the data definition is also a Binary attribute,
        it then returns true, and false otherwise.

        PARAMETERS
        ----------
        instance : Instance
            Instance to checks the attribute type.

        RETURNS
        -------
        bool
            true if attribute types of given Instance and data definition matches.
        """
        for i in range(instance.attributeSize()):
            if isinstance(instance.getAttribute(i), BinaryAttribute):
                if self.__definition.getAttributeType(i) is not AttributeType.BINARY:
                    return False
            elif isinstance(instance.getAttribute(i), DiscreteIndexedAttribute):
                if self.__definition.getAttributeType(i) is not AttributeType.DISCRETE_INDEXED:
                    return False
            elif isinstance(instance.getAttribute(i), DiscreteAttribute):
                if self.__definition.getAttributeType(i) is not AttributeType.DISCRETE:
                    return False
            elif isinstance(instance.getAttribute(i), ContinuousAttribute):
                if self.__definition.getAttributeType(i) is not AttributeType.CONTINUOUS:
                    return False
        return True

    def __setDefinition(self, instance: Instance):
        """
        Adds the attribute types according to given Instance. For instance, if the attribute type of given Instance
        is a Discrete type, it than adds a discrete attribute type to the list of attribute types.

        PARAMETERS
        ----------
        instance : Instance
            Instance input.
        """
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
        self.__definition = DataDefinition(attributeTypes)

    def sampleSize(self) -> int:
        """
        Returns the size of the InstanceList.

        RETURNS
        -------
        int
            Size of the InstanceList.
        """
        return self.__instances.size()

    def classCount(self) -> int:
        """
        Returns the size of the class label distribution of InstanceList.

        RETURNS
        -------
        int
            Size of the class label distribution of InstanceList.
        """
        return len(self.__instances.classDistribution())

    def attributeCount(self) -> int:
        """
        Returns the number of attribute types at DataDefinition list.

        RETURNS
        -------
        int
            The number of attribute types at DataDefinition list.
        """
        return self.__definition.attributeCount()

    def discreteAttributeCount(self) -> int:
        """
        Returns the number of discrete attribute types at DataDefinition list.

        RETURNS
        -------
        int
            The number of discrete attribute types at DataDefinition list.
        """
        return self.__definition.discreteAttributeCount()

    def continuousAttributeCount(self) -> int:
        """
        Returns the number of continuous attribute types at DataDefinition list.

        RETURNS
        -------
        int
            The number of continuous attribute types at DataDefinition list.
        """
        return self.__definition.continuousAttributeCount()

    def getClasses(self) -> str:
        """
        Returns the accumulated String of class labels of the InstanceList.

        RETURNS
        -------
        str
            The accumulated String of class labels of the InstanceList.
        """
        classLabels = self.__instances.getDistinctClassLabels()
        result = classLabels[0]
        for i in range(1, len(classLabels)):
            result = result + ";" + classLabels[i]
        return result

    def info(self, dataSetName: str) -> str:
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
        result = "DATASET: " + dataSetName + "\n"
        result = result + "Number of instances: " + self.sampleSize().__str__() + "\n"
        result = result + "Number of distinct class labels: " + self.classCount().__str__() + "\n"
        result = result + "Number of attributes: " + self.attributeCount().__str__() + "\n"
        result = result + "Number of discrete attributes: " + self.discreteAttributeCount().__str__() + "\n"
        result = result + "Number of continuous attributes: " + self.continuousAttributeCount().__str__() + "\n"
        result = result + "Class labels: " + self.getClasses()
        return result

    def addInstance(self, current: Instance):
        """
        Adds a new instance to the InstanceList.

        PARAMETERS
        ----------
        current : Instance
            Instance to add.
        """
        if self.__definition is None:
            self.__setDefinition(current)
            self.__instances.add(current)
        elif self.__checkDefinition(current):
            self.__instances.add(current)

    def addInstanceList(self, instanceList: list):
        """
        Adds all the instances of given instance list to the InstanceList.

        PARAMETERS
        ----------
        instanceList : list
            InstanceList to add instances from.
        """

        for instance in instanceList:
            self.addInstance(instance)

    def getInstances(self) -> list:
        """
        Returns the instances of InstanceList.

        RETURNS
        -------
        list
            The instances of InstanceList.
        """
        return self.__instances.getInstances()

    def getClassInstances(self) -> list:
        """
        Returns instances of the items at the list of instance lists from the partitions.

        RETURNS
        -------
        list
            Instances of the items at the list of instance lists from the partitions.
        """
        return Partition(self.__instances).getLists()

    def getInstanceList(self) -> InstanceList:
        """
        Accessor for the InstanceList.

        RETURNS
        -------
        InstanceList
            The InstanceList.
        """
        return self.__instances

    def getDataDefinition(self) -> DataDefinition:
        """
        Accessor for the data definition.

        RETURNS
        -------
        DataDefinition
            The data definition.
        """
        return self.__definition

    def getSubSetOfFeatures(self, featureSubSet: FeatureSubSet) -> DataSet:
        """
        Return a subset generated via the given FeatureSubSet.

        PARAMETERS
        ----------
        featureSubSet : FeatureSubSet
            FeatureSubSet input.

        RETURNS
        -------
        FeatureSubSet
            Subset generated via the given FeatureSubSet.
        """
        result = DataSet(self.__definition.getSubSetOfFeatures(featureSubSet))
        for i in range(self.__instances.size()):
            result.addInstance(self.__instances.get(i).getSubSetOfFeatures(featureSubSet))
        return result

    def writeToFile(self, outFileName: str):
        """
        Print out the instances of InstanceList as a String.

        PARAMETERS
        ----------
        outFileName : str
            File name to write the output.
        """
        outfile = open(outFileName, "w")
        for i in range(self.__instances.size()):
            outfile.write(self.__instances.get(i).__str__() + "\n")
        outfile.close()

from __future__ import annotations
import random
import math
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.Instance.Instance import Instance
from Classification.Attribute.AttributeType import AttributeType
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.BinaryAttribute import BinaryAttribute
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Sampling.Bootstrap import Bootstrap
from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.InstanceList.Partition import Partition
from Classification.InstanceList.InstanceListOfSameClass import InstanceListOfSameClass
from Classification.Attribute.Attribute import Attribute
from Classification.Attribute.DiscreteIndexedAttribute import DiscreteIndexedAttribute
from Math.DiscreteDistribution import DiscreteDistribution
from Math.Vector import Vector
from Math.Matrix import Matrix


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
        A list of class labels.
    """
    def getClassLabels(self) -> list:
        classLabels = []
        for instance in self.list:
            classLabels.append(instance.getClassLabel())
        return classLabels

    """
    Extracts the class labels of each instance in the instance list and returns them as a set.

    RETURNS
    -------
    list
        A list of distinct class labels.
    """
    def getDistinctClassLabels(self) -> list:
        classLabels = []
        for instance in self.list:
            if not instance.getClassLabel() in classLabels:
                classLabels.append(instance.getClassLabel())
        return classLabels

    """
    Extracts the possible class labels of each instance in the instance list and returns them as a set.

    RETURNS
    -------
    list
        A list of distinct class labels.
    """
    def getUnionOfPossibleClassLabels(self) -> list:
        possibleClassLabels = []
        for instance in self.list:
            if isinstance(instance, CompositeInstance):
                for possibleClassLabel in instance.getPossibleClassLabels():
                    if not possibleClassLabel in possibleClassLabels:
                        possibleClassLabels.append(possibleClassLabel)
            else:
                if not instance.getClassLabel() in possibleClassLabels:
                    possibleClassLabels.append(instance.getClassLabel())
        return possibleClassLabels

    """
    Divides the instances in the instance list into partitions so that all instances of a class are grouped in a
    single partition.

    RETURNS
    -------
    Partition
        Groups of instances according to their class labels.
    """
    def divideIntoClasses(self) -> Partition:
        classLabels = self.getDistinctClassLabels()
        result = Partition()
        for classLabel in classLabels:
            result.add(InstanceListOfSameClass(classLabel))
        for instance in self.list:
            result.get(classLabels.index(instance.getClassLabel())).add(instance)
        return result

    """
    Extracts distinct discrete values of a given attribute as an array of strings.

    PARAMETERS
    ----------
    attributeIndex : int
        Index of the discrete attribute.
        
    RETURNS
    -------
    list
        An list of distinct values of a discrete attribute.
    """
    def getAttributeValueList(self, attributeIndex: int) -> list:
        valueList = []
        for instance in self.list:
            if not instance.getAttribute(attributeIndex).getValue() in valueList:
                valueList.append(instance.getAttribute(attributeIndex).getValue())
        return valueList

    """
    Creates a stratified partition of the current instance list. In a stratified partition, the percentage of each
    class is preserved. For example, let's say there are three classes in the instance list, and let the percentages of
    these classes be %20, %30, and %50; then the percentages of these classes in the stratified partitions are the
    same, that is, %20, %30, and %50.

    PARAMETERS
    ----------
    ratio : float
        Ratio of the stratified partition. Ratio is between 0 and 1. If the ratio is 0.2, then 20 percent of the 
        instances are put in the first group, 80 percent of the instances are put in the second group.
    seed : int
        seed is used as a random number.
        
    RETURNS
    -------
    Partition
        2 group stratified partition of the instances in this instance list.
    """
    def stratifiedPartition(self, ratio: float, seed: int) -> Partition:
        partition = Partition()
        partition.add(InstanceList())
        partition.add(InstanceList())
        distribution = self.classDistribution()
        counts = [0] * distribution.size()
        randomArray = [i for i in range(self.size())]
        random.shuffle(randomArray, seed)
        for i in range(self.size()):
            instance = self.list[randomArray[i]]
            classIndex = distribution.getIndex(instance.getClassLabel())
            if counts[classIndex] < self.size() * ratio * distribution.getProbability(instance.getClassLabel()):
                partition.get(0).add(instance)
            else:
                partition.get(1).add(instance)
            counts[classIndex] = counts[classIndex] + 1
        return partition

    """
    Creates a partition of the current instance list.

    PARAMETERS
    ----------
    ratio : float
        Ratio of the stratified partition. Ratio is between 0 and 1. If the ratio is 0.2, then 20 percent of the 
        instances are put in the first group, 80 percent of the instances are put in the second group.
    seed : int
        seed is used as a random number.

    RETURNS
    -------
    Partition
        2 group stratified partition of the instances in this instance list.
    """
    def partition(self, ratio: float, seed: int) -> Partition:
        partition = Partition()
        partition.add(InstanceList())
        partition.add(InstanceList())
        random.shuffle(self.list, seed)
        for i in range(self.size()):
            instance = self.list[i]
            if i < self.size() * ratio:
                partition.get(0).add(instance)
            else:
                partition.get(1).add(instance)
        return partition

    """
    Creates a partition depending on the distinct values of a discrete attribute. If the discrete attribute has 4
    distinct values, the resulting partition will have 4 groups, where each group contain instance whose
    values of that discrete attribute are the same.

    PARAMETERS
    ----------
    attributeIndex : int
        Index of the discrete attribute.
        
    RETURNS
    -------
    Partition
        L groups of instances, where L is the number of distinct values of the discrete attribute with index 
        attributeIndex.
    """
    def divideWithRespectToDiscreteAttribute(self, attributeIndex: int) -> Partition:
        valueList = self.getAttributeValueList(attributeIndex)
        result = Partition()
        for value in valueList:
            result.add(InstanceList())
        for instance in self.list:
            result.get(valueList.index(instance.getAttribute(attributeIndex).getValue())).add(instance)
        return result

    """
    Creates a partition depending on the distinct values of a discrete indexed attribute.

    PARAMETERS
    ----------
    attributeIndex : int
        Index of the discrete indexed attribute.
    attributeValue : int
        Value of the attribute.
    
    RETURNS
    -------
    Partition
        L groups of instances, where L is the number of distinct values of the discrete indexed attribute with index 
        attributeIndex and value attributeValue.
    """
    def divideWithRespectToIndexedAtribute(self, attributeIndex: int, attributeValue: int) -> Partition:
        result = Partition()
        result.add(InstanceList())
        result.add(InstanceList())
        for instance in self.list:
            if instance.getAttribute(attributeIndex).getIndex() == attributeIndex:
                result.get(0).add(instance)
            else:
                result.get(1).add(instance)
        return result

    """
    Creates a two group partition depending on the values of a continuous attribute. If the value of the attribute is
    less than splitValue, the instance is forwarded to the first group, else it is forwarded to the second group.

    PARAMETERS
    ----------
    attributeIndex : int
        Index of the discrete indexed attribute.
    splitValue : float    
        Threshold to divide instances
        
    RETURNS
    -------
    Partition
        Two groups of instances as a partition.
    """
    def divideWithRespectToContinuousAttribute(self, attributeIndex: int, splitValue: float) -> Partition:
        result = Partition()
        result.add(InstanceList())
        result.add(InstanceList())
        for instance in self.list:
            if instance.getAttribute(attributeIndex).getValue() < splitValue:
                result.get(0).add(instance)
            else:
                result.get(1).add(instance)
        return result

    """
    Calculates the mean of a single attribute for this instance list (m_i). If the attribute is discrete, the maximum
    occurring value for that attribute is returned. If the attribute is continuous, the mean value of the values of
    all instances are returned.

    PARAMETERS
    ----------
    index : int
        Index of the attribute.
        
    RETURNS
    -------
    Attribute
        The mean value of the instances as an attribute.
    """
    def attributeAverage(self, index: int) -> Attribute:
        if isinstance(self.list[0].getAttribute(index), DiscreteAttribute):
            values = []
            for instance in self.list:
                values.append(instance.getAttribute(index).getValue())
            return DiscreteAttribute(Classifier.getMaximum(values))
        elif isinstance(self.list[0].getAttribute(index), ContinuousAttribute):
            sum = 0.0
            for instance in self.list:
                sum += instance.getAttribute(index).getValue()
            return ContinuousAttribute(sum / len(self.list))
        else:
            return None

    """
    Calculates the mean of a single attribute for this instance list (m_i).

    PARAMETERS
    ----------
    index : int
        Index of the attribute.
        
    RETURNS
    -------
    list
        The mean value of the instances as an attribute.
    """
    def continuousAttributeAverage(self, index: int) -> list:
        if isinstance(self.list[0].getAttribute(index), DiscreteIndexedAttribute):
            maxIndexSize = self.list[0].getAttribute(index).getMaxIndex()
            values = [0.0] * maxIndexSize
            for instance in self.list:
                valueIndex = instance.getAttribute(index).getIndex()
                values[valueIndex] = values[valueIndex] + 1
            for i in range(len(values)):
                values[i] = values[i] / len(self.list)
            return values
        elif isinstance(self.list[0].getAttribute(index), ContinuousAttribute):
            sum = 0.0
            for instance in self.list:
                sum += instance.getAttribute(index).getValue()
            return [sum / len(self.list)]
        else:
            return None

    """
    Calculates the standard deviation of a single attribute for this instance list (m_i). If the attribute is discrete,
    None returned. If the attribute is continuous, the standard deviation  of the values all instances are returned.

    PARAMETERS
    ----------
    index : int
        Index of the attribute.
        
    RETURNS
    -------
    Attribute
        The standard deviation of the instances as an attribute.
    """
    def attributeStandardDeviation(self, index: int) -> Attribute:
        if isinstance(self.list[0].getAttribute(index), ContinuousAttribute):
            sum = 0.0
            for instance in self.list:
                sum += instance.getAttribute(index).getValue()
            average = sum / len(self.list)
            sum = 0.0
            for instance in self.list:
                sum += math.pow(instance.getAttribute(index).getValue() - average, 2)
            return ContinuousAttribute(math.sqrt(sum / (len(self.list) - 1)))
        else:
            return None

    """
    Calculates the standard deviation of a single continuous attribute for this instance list (m_i).

    PARAMETERS
    ----------
    index : int
        Index of the attribute.
        
    RETURNS
    -------
    list
        The standard deviation of the instances as an attribute.
    """
    def continuousAttributeStandardDeviation(self, index: int) -> list:
        if isinstance(self.list[0].getAttribute(index), DiscreteIndexedAttribute):
            maxIndexSize = self.list[0].getAttribute(index).getMaxIndex()
            averages = [0.0] * maxIndexSize
            for instance in self.list:
                valueIndex = instance.getAttribute(index).getIndex()
                averages[valueIndex] = averages[valueIndex] + 1
            for i in range(len(averages)):
                averages[i] = averages[i] / len(self.list)
            values = [0.0] * maxIndexSize
            for instance in self.list:
                valueIndex = instance.getAttribute(index).getIndex()
                for i in range(maxIndexSize):
                    if i == valueIndex:
                        values[i] += math.pow(1 - averages[i], 2)
                    else:
                        values[i] += math.pow(averages[i], 2)
            for i in range(len(values)):
                values[i] = math.sqrt(values[i] / (len(self.list) - 1))
            return values
        elif isinstance(self.list[0].getAttribute(index), ContinuousAttribute):
            sum = 0.0
            for instance in self.list:
                sum += instance.getAttribute(index).getValue()
            average = sum / len(self.list)
            for instance in self.list:
                sum += math.pow(instance.getAttribute(index).getValue() - average, 2)
            return [math.sqrt(sum / (len(self.list) - 1))]
        else:
            return None

    """
    The attributeDistribution method takes an index as an input and if the attribute of the instance at given index is
    discrete, it returns the distribution of the attributes of that instance.

    PARAMETERS
    ----------
    index : int
        Index of the attribute.
        
    RETURNS
    -------
    DiscreteDistribution
        Distribution of the attribute.
    """
    def attributeDistribution(self, index: int) -> DiscreteDistribution:
        distribution = DiscreteDistribution()
        if isinstance(self.list[0].getAttribute(index), DiscreteAttribute):
            for instance in self.list:
                distribution.addItem(instance.getAttribute(index).getValue())
        return distribution

    """
    The attributeClassDistribution method takes an attribute index as an input. It loops through the instances, gets
    the corresponding value of given attribute index and adds the class label of that instance to the discrete 
    distributions list.

    PARAMETERS
    ----------
    attributeIndex : int
        Index of the attribute.
        
    RETURNS
    -------
    list
        Distribution of the class labels.
    """
    def attributeClassDistribution(self, attributeIndex: int) -> list:
        distributions = []
        valueList = self.getAttributeValueList(attributeIndex)
        for ignored in valueList:
            distributions.append(DiscreteDistribution())
        for instance in self.list:
            distributions[valueList.index(instance.getAttribute(attributeIndex).getValue())].addItem(instance.getClassLabel())
        return distributions

    """
    The discreteIndexedAttributeClassDistribution method takes an attribute index and an attribute value as inputs.
    It loops through the instances, gets the corresponding value of given attribute index and given attribute value.
    Then, adds the class label of that instance to the discrete indexed distributions list.

    PARAMETERS
    ----------
    attributeIndex : int
        Index of the attribute.
    attributeValue : int
        Value of the attribute.
        
    RETURNS
    -------
    DiscreteDistribution
        Distribution of the class labels.
    """
    def discreteIndexedAttributeClassDistribution(self, attributeIndex: int, attributeValue: int) -> DiscreteDistribution:
        distribution = DiscreteDistribution()
        for instance in self.list:
            if instance.getAttribute(attributeIndex).getIndex() == attributeValue:
                distribution.addItem(instance.getClassLabel())
        return distribution

    """
    The classDistribution method returns the distribution of all the class labels of instances.

    RETURNS
    -------
    DiscreteDistribution
        Distribution of the class labels.
    """
    def classDistribution(self) -> DiscreteDistribution:
        distribution = DiscreteDistribution()
        for instance in self.list:
            distribution.addItem(instance.getClassLabel())
        return distribution

    """
    The allAttributesDistribution method returns the distributions of all the attributes of instances.

    RETURNS
    -------
    list
        Distributions of all the attributes of instances.
    """
    def allAttributesDistribution(self) -> list:
        distributions = []
        for i in range(self.list[0].attributeSize()):
            distributions.append(self.attributeDistribution(i))
        return distributions

    """
    Returns the mean of all the attributes for instances in the list.

    RETURNS
    -------
    Instance
        Mean of all the attributes for instances in the list.
    """
    def average(self) -> Instance:
        result = Instance(list[0].getClassLabel())
        for i in range(self.list[0].attributeSize()):
            result.addAttribute(self.attributeAverage(i))
        return result

    """
    Calculates mean of the attributes of instances.

    RETURNS
    -------
    list
        Mean of the attributes of instances.
    """
    def continuousAverage(self) -> list:
        result = []
        for i in range(self.list[0].attributeSize()):
            result.extend(self.continuousAttributeAverage(i))
        return result

    """
    Returns the standard deviation of attributes for instances.

    RETURNS
    -------
    Instance
        Standard deviation of attributes for instances.
    """
    def standardDeviation(self) -> Instance:
        result = Instance(list[0].getClassLabel())
        for i in range(self.list[0].attributeSize()):
            result.addAttribute(self.attributeStandardDeviation(i))
        return result

    """
    Returns the standard deviation of continuous attributes for instances.

    RETURNS
    -------
    list
        Standard deviation of continuous attributes for instances.
    """
    def continuousStandardDeviation(self) -> list:
        result = []
        for i in range(self.list[0].attributeSize()):
            result.extend(self.continuousAttributeStandardDeviation(i))
        return result

    """
    Calculates a covariance Matrix by using an average Vector.

    PARAMETERS
    ----------
    average : Vector
        Vector input.
        
    RETURNS
    -------
    Matrix
        Covariance Matrix.
    """
    def covariance(self, average: Vector) -> Matrix:
        result = Matrix(list[0].continuousAttributeSize(), list[0].continuousAttributeSize())
        for instance in self.list:
            continuousAttributes = instance.continuousAttributes()
            for i in range(instance.continuousAttributeSize()):
                xi = continuousAttributes[i]
                mi = average.getValue(i)
                for j in range(instance.continuousAttributeSize()):
                    xj = continuousAttributes[j]
                    mj = average.getValue(j)
                    result.addValue(i, j, (xi - mi) * (xj - mj))
        result.divideByConstant(len(self.list) - 1)
        return result

    """
    Accessor for the instances.

    RETURNS
    -------
    list
        Instances.
    """
    def getInstances(self) -> list:
        return self.list

from __future__ import annotations
import random
import math
from functools import cmp_to_key

from Classification.DataSet.DataDefinition import DataDefinition
from Classification.Instance.Instance import Instance
from Classification.Attribute.AttributeType import AttributeType
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.BinaryAttribute import BinaryAttribute
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Sampling.Bootstrap import Bootstrap
from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Attribute.Attribute import Attribute
from Classification.Attribute.DiscreteIndexedAttribute import DiscreteIndexedAttribute
from Math.DiscreteDistribution import DiscreteDistribution
from Math.Vector import Vector
from Math.Matrix import Matrix

from Classification.Model.Model import Model


class InstanceList(object):

    list: list

    def __init__(self, listOrDefinition = None, separator: str = None, fileName: str = None):
        """
        Constructor for an instance list with a given data definition, data file and a separator character. Each
        instance must be stored in a separate line separated with the character separator. The last item must be the
        class label. The function reads the file line by line and for each line; depending on the data definition, that
        is, type of the attributes, adds discrete and continuous attributes to a new instance. For example, given the
        data set file

        red;1;0.4;true
        green;-1;0.8;true
        blue;3;1.3;false

        where the first attribute is a discrete attribute, second and third attributes are continuous attributes, the
        fourth item is the class label.

        PARAMETERS
        ----------
        listOrDefinition
            Data definition of the data set.
        separator : str
            Separator character which separates the attribute values in the data file.
        fileName : str
            Name of the data set file.
        """
        if listOrDefinition is None:
            self.list = []
        else:
            if separator is None and isinstance(listOrDefinition, list):
                self.list = listOrDefinition
            else:
                if isinstance(listOrDefinition, DataDefinition):
                    self.list = []
                    file = open(fileName, 'r', encoding='utf8')
                    lines = file.readlines()
                    for line in lines:
                        attributeList = line.strip().split(separator)
                        if len(attributeList) == listOrDefinition.attributeCount() + 1:
                            current = Instance(attributeList[len(attributeList) - 1])
                            for i in range(len(attributeList) - 1):
                                if listOrDefinition.getAttributeType(i) is AttributeType.DISCRETE:
                                    current.addAttribute(DiscreteAttribute(attributeList[i]))
                                elif listOrDefinition.getAttributeType(i) is AttributeType.BINARY:
                                    current.addAttribute(
                                        BinaryAttribute(attributeList[i] in ["True", "true", "Yes", "yes", "y", "Y"]))
                                elif listOrDefinition.getAttributeType(i) is AttributeType.CONTINUOUS:
                                    current.addAttribute(ContinuousAttribute(float(attributeList[i])))
                            self.list.append(current)

    def add(self, instance: Instance):
        """
        Adds instance to the instance list.

        PARAMETERS
        ----------
        instance : Instance
            Instance to be added.
        """
        self.list.append(instance)

    def addAll(self, instanceList: list):
        """
        Adds a list of instances to the current instance list.

        PARAMETERS
        ----------
        instanceList : list
            List of instances to be added.
        """
        self.list.extend(instanceList)

    def size(self) -> int:
        """
        Returns size of the instance list.

        RETURNS
        -------
        int
            Size of the instance list.
        """
        return len(self.list)

    def get(self, index: int) -> Instance:
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
        return self.list[index]

    def makeComparator(self, attributeIndex: int):
        def compare(instanceA: Instance, instanceB: Instance):
            result1 = instanceA.getAttribute(attributeIndex).getValue()
            result2 = instanceB.getAttribute(attributeIndex).getValue()
            if result1 < result2:
                return -1
            elif result1 > result2:
                return 1
            else:
                return 0
        return compare

    def sortWrtAttribute(self, attributeIndex: int):
        """
        Sorts attribute list according to the attribute with index 'attributeIndex'.

        PARAMETERS
        ----------
        attributeIndex : int
            index of the attribute.
        """
        self.list.sort(key=cmp_to_key(self.makeComparator(attributeIndex)))

    def sort(self):
        """
        Sorts attributes list.
        """
        self.list.sort()

    def shuffle(self, seed: int):
        """
        Shuffles the instance list.

        PARAMETERS
        ----------
        seed : int
            Seed is used for random number generation.
        """
        random.seed(seed)
        random.shuffle(self.list)

    def bootstrap(self, seed: int) -> Bootstrap:
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
        return Bootstrap(self.list, seed)

    def getClassLabels(self) -> list:
        """
        Extracts the class labels of each instance in the instance list and returns them in an array of {@link String}.

        RETURNS
        -------
        list
            A list of class labels.
        """
        classLabels = []
        for instance in self.list:
            classLabels.append(instance.getClassLabel())
        return classLabels

    def getDistinctClassLabels(self) -> list:
        """
        Extracts the class labels of each instance in the instance list and returns them as a set.

        RETURNS
        -------
        list
            A list of distinct class labels.
        """
        classLabels = []
        for instance in self.list:
            if not instance.getClassLabel() in classLabels:
                classLabels.append(instance.getClassLabel())
        return classLabels

    def getUnionOfPossibleClassLabels(self) -> list:
        """
        Extracts the possible class labels of each instance in the instance list and returns them as a set.

        RETURNS
        -------
        list
            A list of distinct class labels.
        """
        possibleClassLabels = []
        for instance in self.list:
            if isinstance(instance, CompositeInstance):
                for possibleClassLabel in instance.getPossibleClassLabels():
                    if possibleClassLabel not in possibleClassLabels:
                        possibleClassLabels.append(possibleClassLabel)
            else:
                if not instance.getClassLabel() in possibleClassLabels:
                    possibleClassLabels.append(instance.getClassLabel())
        return possibleClassLabels

    def getAttributeValueList(self, attributeIndex: int) -> list:
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
        valueList = []
        for instance in self.list:
            if not instance.getAttribute(attributeIndex).getValue() in valueList:
                valueList.append(instance.getAttribute(attributeIndex).getValue())
        return valueList

    def __attributeAverage(self, index: int) -> Attribute:
        """
        Calculates the mean of a single attribute for this instance list (m_i). If the attribute is discrete, the
        maximum occurring value for that attribute is returned. If the attribute is continuous, the mean value of the
        values of all instances are returned.

        PARAMETERS
        ----------
        index : int
            Index of the attribute.

        RETURNS
        -------
        Attribute
            The mean value of the instances as an attribute.
        """
        if isinstance(self.list[0].getAttribute(index), DiscreteAttribute):
            values = []
            for instance in self.list:
                values.append(instance.getAttribute(index).getValue())
            return DiscreteAttribute(Model.getMaximum(values))
        elif isinstance(self.list[0].getAttribute(index), ContinuousAttribute):
            total = 0.0
            for instance in self.list:
                total += instance.getAttribute(index).getValue()
            return ContinuousAttribute(total / len(self.list))
        else:
            return None

    def continuousAttributeAverage(self, index: int) -> list:
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
            total = 0.0
            for instance in self.list:
                total += instance.getAttribute(index).getValue()
            return [total / len(self.list)]
        else:
            return None

    def __attributeStandardDeviation(self, index: int) -> Attribute:
        """
        Calculates the standard deviation of a single attribute for this instance list (m_i). If the attribute is
        discrete, None returned. If the attribute is continuous, the standard deviation  of the values all instances are
        returned.

        PARAMETERS
        ----------
        index : int
            Index of the attribute.

        RETURNS
        -------
        Attribute
            The standard deviation of the instances as an attribute.
        """
        if isinstance(self.list[0].getAttribute(index), ContinuousAttribute):
            total = 0.0
            for instance in self.list:
                total += instance.getAttribute(index).getValue()
            average = total / len(self.list)
            total = 0.0
            for instance in self.list:
                total += math.pow(instance.getAttribute(index).getValue() - average, 2)
            return ContinuousAttribute(math.sqrt(total / (len(self.list) - 1)))
        else:
            return None

    def continuousAttributeStandardDeviation(self, index: int) -> list:
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
            total = 0.0
            for instance in self.list:
                total += instance.getAttribute(index).getValue()
            average = total / len(self.list)
            for instance in self.list:
                total += math.pow(instance.getAttribute(index).getValue() - average, 2)
            return [math.sqrt(total / (len(self.list) - 1))]
        else:
            return None

    def attributeDistribution(self, index: int) -> DiscreteDistribution:
        """
        The attributeDistribution method takes an index as an input and if the attribute of the instance at given index
        is discrete, it returns the distribution of the attributes of that instance.

        PARAMETERS
        ----------
        index : int
            Index of the attribute.

        RETURNS
        -------
        DiscreteDistribution
            Distribution of the attribute.
        """
        distribution = DiscreteDistribution()
        if isinstance(self.list[0].getAttribute(index), DiscreteAttribute):
            for instance in self.list:
                distribution.addItem(instance.getAttribute(index).getValue())
        return distribution

    def attributeClassDistribution(self, attributeIndex: int) -> list:
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
        distributions = []
        valueList = self.getAttributeValueList(attributeIndex)
        for _ in valueList:
            distributions.append(DiscreteDistribution())
        for instance in self.list:
            distributions[valueList.index(instance.getAttribute(attributeIndex).getValue())].addItem(instance.
                                                                                                     getClassLabel())
        return distributions

    def discreteIndexedAttributeClassDistribution(self, attributeIndex: int, attributeValue: int) -> \
            DiscreteDistribution:
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
        distribution = DiscreteDistribution()
        for instance in self.list:
            if instance.getAttribute(attributeIndex).getIndex() == attributeValue:
                distribution.addItem(instance.getClassLabel())
        return distribution

    def classDistribution(self) -> DiscreteDistribution:
        """
        The classDistribution method returns the distribution of all the class labels of instances.

        RETURNS
        -------
        DiscreteDistribution
            Distribution of the class labels.
        """
        distribution = DiscreteDistribution()
        for instance in self.list:
            distribution.addItem(instance.getClassLabel())
        return distribution

    def allAttributesDistribution(self) -> list:
        """
        The allAttributesDistribution method returns the distributions of all the attributes of instances.

        RETURNS
        -------
        list
            Distributions of all the attributes of instances.
        """
        distributions = []
        for i in range(self.list[0].attributeSize()):
            distributions.append(self.attributeDistribution(i))
        return distributions

    def average(self) -> Instance:
        """
        Returns the mean of all the attributes for instances in the list.

        RETURNS
        -------
        Instance
            Mean of all the attributes for instances in the list.
        """
        result = Instance(self.list[0].getClassLabel())
        for i in range(self.list[0].attributeSize()):
            result.addAttribute(self.__attributeAverage(i))
        return result

    def continuousAverage(self) -> list:
        """
        Calculates mean of the attributes of instances.

        RETURNS
        -------
        list
            Mean of the attributes of instances.
        """
        result = []
        for i in range(self.list[0].attributeSize()):
            result.extend(self.continuousAttributeAverage(i))
        return result

    def standardDeviation(self) -> Instance:
        """
        Returns the standard deviation of attributes for instances.

        RETURNS
        -------
        Instance
            Standard deviation of attributes for instances.
        """
        result = Instance(self.list[0].getClassLabel())
        for i in range(self.list[0].attributeSize()):
            result.addAttribute(self.__attributeStandardDeviation(i))
        return result

    def continuousStandardDeviation(self) -> list:
        """
        Returns the standard deviation of continuous attributes for instances.

        RETURNS
        -------
        list
            Standard deviation of continuous attributes for instances.
        """
        result = []
        for i in range(self.list[0].attributeSize()):
            result.extend(self.continuousAttributeStandardDeviation(i))
        return result

    def covariance(self, average: Vector) -> Matrix:
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
        result = Matrix(self.list[0].continuousAttributeSize(), self.list[0].continuousAttributeSize())
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

    def getInstances(self) -> list:
        """
        Accessor for the instances.

        RETURNS
        -------
        list
            Instances.
        """
        return self.list

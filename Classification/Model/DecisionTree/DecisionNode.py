from Math.DiscreteDistribution import DiscreteDistribution
from Util.RandomArray import RandomArray

from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.DiscreteIndexedAttribute import DiscreteIndexedAttribute
from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.DecisionTree.DecisionCondition import DecisionCondition
from Classification.Model.Model import Model
from Classification.Parameter.RandomForestParameter import RandomForestParameter
import random


class DecisionNode(object):
    children: list
    __data: InstanceList
    __class_label: str
    leaf: bool
    __condition: DecisionCondition
    EPSILON = 0.0000000001

    def __init__(self,
                 data: InstanceList,
                 condition=None,
                 parameter=None,
                 isStump=False):
        """
        The DecisionNode method takes InstanceList data as input and then it sets the class label parameter by finding
        the most occurred class label of given data, it then gets distinct class labels as class labels ArrayList.
        Later, it adds ordered indices to the indexList and shuffles them randomly. Then, it gets the class distribution
        of given data and finds the best entropy value of these class distribution.

        If an attribute of given data is DiscreteIndexedAttribute, it creates a Distribution according to discrete
        indexed attribute class distribution and finds the entropy. If it is better than the last best entropy it
        reassigns the best entropy, best attribute and best split value according to the newly founded best entropy's
        index. At the end, it also add new distribution to the class distribution.

        If an attribute of given data is DiscreteAttribute, it directly finds the entropy. If it is better than the last
        best entropy it reassigns the best entropy, best attribute and best split value according to the newly founded
        best entropy's index.

        If an attribute of given data is ContinuousAttribute, it creates two distributions; left and right according
        to class distribution and discrete distribution respectively, and finds the entropy. If it is better than the
        last best entropy it reassigns the best entropy, best attribute and best split value according to the newly
        founded best entropy's index. At the end, it also add new distribution to the right distribution and removes
        from left distribution.

        PARAMETERS
        ----------
        data : InstanceList
            InstanceList input.
        condition : DecisionCondition
            DecisionCondition to check.
        parameter : RandomForestParameter
            RandomForestParameter like seed, ensembleSize, attributeSubsetSize.
        isStump : bool
            Refers to decision trees with only 1 splitting rule.
        """
        best_attribute = -1
        best_split_value = 0
        self.__condition = condition
        self.__data = data
        self.__class_label = Model.getMaximum(self.__data.getClassLabels())
        self.leaf = True
        self.children = []
        class_labels = self.__data.getDistinctClassLabels()
        if len(class_labels) == 1:
            return
        if isStump and condition is not None:
            return
        if parameter is not None and parameter.getAttributeSubsetSize() < data.get(0).attributeSize():
            index_list = RandomArray.indexArray(data.get(0).attributeSize(), parameter.getSeed())
            size = parameter.getAttributeSubsetSize()
        else:
            index_list = [i for i in range(data.get(0).attributeSize())]
            size = data.get(0).attributeSize()
        class_distribution = data.classDistribution()
        best_entropy = data.classDistribution().entropy()
        for j in range(size):
            index = index_list[j]
            if isinstance(data.get(0).getAttribute(index), DiscreteIndexedAttribute):
                for k in range(data.get(0).getAttribute(index).getMaxIndex()):
                    distribution = data.discreteIndexedAttributeClassDistribution(index, k)
                    if distribution.getSum() > 0:
                        class_distribution.removeDistribution(distribution)
                        entropy = (
                                              class_distribution.entropy() * class_distribution.getSum() + distribution.entropy() * distribution.getSum()) / data.size()
                        if entropy + self.EPSILON < best_entropy:
                            best_entropy = entropy
                            best_attribute = index
                            best_split_value = k
                        class_distribution.addDistribution(distribution)
            elif isinstance(data.get(0).getAttribute(index), DiscreteAttribute):
                entropy = self.__entropyForDiscreteAttribute(index)
                if entropy + self.EPSILON < best_entropy:
                    best_entropy = entropy
                    best_attribute = index
            elif isinstance(data.get(0).getAttribute(index), ContinuousAttribute):
                data.sortWrtAttribute(index)
                previous_value = -100000000
                left_distribution = data.classDistribution()
                right_distribution = DiscreteDistribution()
                for k in range(data.size()):
                    instance = data.get(k)
                    if k == 0:
                        previous_value = instance.getAttribute(index).getValue()
                    elif instance.getAttribute(index).getValue() != previous_value:
                        split_value = (previous_value + instance.getAttribute(index).getValue()) / 2
                        previous_value = instance.getAttribute(index).getValue()
                        entropy = (left_distribution.getSum() / data.size()) * left_distribution.entropy() + \
                                  (right_distribution.getSum() / data.size()) * right_distribution.entropy()
                        if entropy + self.EPSILON < best_entropy:
                            best_entropy = entropy
                            best_split_value = split_value
                            best_attribute = index
                    left_distribution.removeItem(instance.getClassLabel())
                    right_distribution.addItem(instance.getClassLabel())
        if best_attribute != -1:
            self.leaf = False
            if isinstance(data.get(0).getAttribute(best_attribute), DiscreteIndexedAttribute):
                self.__createChildrenForDiscreteIndexed(attributeIndex=best_attribute,
                                                        attributeValue=best_split_value,
                                                        parameter=parameter,
                                                        isStump=isStump)
            elif isinstance(data.get(0).getAttribute(best_attribute), DiscreteAttribute):
                self.__createChildrenForDiscrete(attributeIndex=best_attribute,
                                                 parameter=parameter,
                                                 isStump=isStump)
            elif isinstance(data.get(0).getAttribute(best_attribute), ContinuousAttribute):
                self.__createChildrenForContinuous(attributeIndex=best_attribute,
                                                   splitValue=best_split_value,
                                                   parameter=parameter,
                                                   isStump=isStump)

    def __entropyForDiscreteAttribute(self, attributeIndex: int):
        """
        The entropyForDiscreteAttribute method takes an attributeIndex and creates an ArrayList of DiscreteDistribution.
        Then loops through the distributions and calculates the total entropy.

        PARAMETERS
        ----------
        attributeIndex : int
            Index of the attribute.

        RETURNS
        -------
        float
            Total entropy for the discrete attribute.
        """
        total = 0.0
        distributions = self.__data.attributeClassDistribution(attributeIndex)
        for distribution in distributions:
            total += (distribution.getSum() / self.__data.size()) * distribution.entropy()
        return total

    def __createChildrenForDiscreteIndexed(self,
                                           attributeIndex: int,
                                           attributeValue: int,
                                           parameter: RandomForestParameter,
                                           isStump: bool):
        """
        The createChildrenForDiscreteIndexed method creates an list of DecisionNodes as children and a partition with
        respect to indexed attribute.

        PARAMETERS
        ----------
        attributeIndex : int
            Index of the attribute.
        attributeValue : int
            Value of the attribute.
        parameter : RandomForestParameter
            Like seed, ensembleSize, attributeSubsetSize.
        isStump : bool
            Refers to decision trees with only 1 splitting rule.
        """
        children_data = Partition(self.__data, attributeIndex, attributeValue)
        self.children.append(
            DecisionNode(data=children_data.get(0),
                         condition=DecisionCondition(attributeIndex,
                                                     DiscreteIndexedAttribute("",
                                                                              attributeValue,
                                                                              self.__data.get(0).getAttribute(
                                                                                  attributeIndex).getMaxIndex())),
                         parameter=parameter,
                         isStump=isStump))
        self.children.append(
            DecisionNode(data=children_data.get(1),
                         condition=DecisionCondition(attributeIndex,
                                                     DiscreteIndexedAttribute("",
                                                                              -1,
                                                                              self.__data.get(0).getAttribute(
                                                                                  attributeIndex).getMaxIndex())),
                         parameter=parameter,
                         isStump=isStump))

    def __createChildrenForDiscrete(self,
                                    attributeIndex: int,
                                    parameter: RandomForestParameter,
                                    isStump: bool):
        """
        The createChildrenForDiscrete method creates an ArrayList of values, a partition with respect to attributes and
        a list of DecisionNodes as children.

        PARAMETERS
        ----------
        attributeIndex : int
            Index of the attribute.
        parameter : RandomForestParameter
            RandomForestParameter like seed, ensembleSize, attributeSubsetSize.
        isStump : bool
            Refers to decision trees with only 1 splitting rule.
        """
        value_list = self.__data.getAttributeValueList(attributeIndex)
        children_data = Partition(self.__data, attributeIndex)
        for i in range(len(value_list)):
            self.children.append(DecisionNode(data=children_data.get(i),
                                              condition=DecisionCondition(attributeIndex=attributeIndex,
                                                                          value=DiscreteAttribute(value_list[i])),
                                              parameter=parameter,
                                              isStump=isStump))

    def __createChildrenForContinuous(self, attributeIndex: int, splitValue: float, parameter: RandomForestParameter,
                                      isStump: bool):
        """
        The createChildrenForContinuous method creates a list of DecisionNodes as children and a partition with respect
        to continuous attribute and the given split value.

        PARAMETERS
        ----------
        attributeIndex : int
            Index of the attribute.
        parameter : RandomForestParameter
            RandomForestParameter like seed, ensembleSize, attributeSubsetSize.
        isStump : bool
            Refers to decision trees with only 1 splitting rule.
        splitValue : float
            Split value is used for partitioning.
        """
        children_data = Partition(self.__data, attributeIndex, splitValue)
        self.children.append(DecisionNode(children_data.get(0),
                                          DecisionCondition(attributeIndex, ContinuousAttribute(splitValue), "<"),
                                          parameter, isStump))
        self.children.append(DecisionNode(children_data.get(1),
                                          DecisionCondition(attributeIndex, ContinuousAttribute(splitValue), ">"),
                                          parameter, isStump))

    def predict(self, instance: Instance) -> str:
        """
        The predict method takes an Instance as input and performs prediction on the DecisionNodes and returns the
        prediction for that instance.

        PARAMETERS
        ----------
        instance : Instance
            Instance to make prediction.

        RETURNS
        -------
        str
            The prediction for given instance.
        """
        if isinstance(instance, CompositeInstance):
            possible_class_labels = instance.getPossibleClassLabels()
            distribution = self.__data.classDistribution()
            predicted_class = distribution.getMaxItemIncludeTheseOnly(possible_class_labels)
            if self.leaf:
                return predicted_class
            else:
                for node in self.children:
                    if node.__condition.satisfy(instance):
                        child_prediction = node.predict(instance)
                        if child_prediction is not None:
                            return child_prediction
                        else:
                            return predicted_class
                return predicted_class
        elif self.leaf:
            return self.__class_label
        else:
            for node in self.children:
                if node.__condition.satisfy(instance):
                    return node.predict(instance)
            return self.__class_label

    def predictProbabilityDistribution(self, instance: Instance) -> dict:
        if self.leaf:
            return self.__data.classDistribution().getProbabilityDistribution()
        else:
            for node in self.children:
                if node.__condition.satisfy(instance):
                    return node.predictProbabilityDistribution(instance)
            return self.__data.classDistribution().getProbabilityDistribution()

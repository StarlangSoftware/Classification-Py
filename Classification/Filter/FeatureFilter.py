from abc import abstractmethod

from Classification.DataSet.DataSet import DataSet
from Classification.Instance.Instance import Instance


class FeatureFilter(object):

    @abstractmethod
    def convertInstance(self, instance: Instance):
        pass

    @abstractmethod
    def convertDataDefinition(self):
        pass

    """
    Constructor that sets the dataSet.

    PARAMETERS
    ----------
    dataSet : DataSet
        DataSet that will be used.
    """
    def __init__(self, dataSet: DataSet):
        self.dataSet = dataSet

    """
    Feature converter for a list of instances. Using the abstract method convertInstance, each instance in the
    instance list will be converted.
    """
    def convert(self):
        instances = self.dataSet.getInstances()
        for instance in instances:
            self.convertInstance(instance)
        self.convertDataDefinition()

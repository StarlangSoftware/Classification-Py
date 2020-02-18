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

    def __init__(self, dataSet: DataSet):
        """
        Constructor that sets the dataSet.

        PARAMETERS
        ----------
        dataSet : DataSet
            DataSet that will be used.
        """
        self.dataSet = dataSet

    def convert(self):
        """
        Feature converter for a list of instances. Using the abstract method convertInstance, each instance in the
        instance list will be converted.
        """
        instances = self.dataSet.getInstances()
        for instance in instances:
            self.convertInstance(instance)
        self.convertDataDefinition()

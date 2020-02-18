from abc import abstractmethod

from Classification.DataSet.DataSet import DataSet
from Classification.Filter.FeatureFilter import FeatureFilter


class TrainedFeatureFilter(FeatureFilter):

    @abstractmethod
    def train(self):
        pass

    def __init__(self, dataSet: DataSet):
        """
        Constructor that sets the dataSet.

        PARAMETERS
        ----------
        dataSet : DataSet
            DataSet that will bu used.
        """
        super().__init__(dataSet)

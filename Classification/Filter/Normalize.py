from Classification.DataSet.DataSet import DataSet
from Classification.Filter.FeatureFilter import FeatureFilter


class Normalize(FeatureFilter):

    """
    Constructor for normalize feature filter. It calculates and stores the mean (m) and standard deviation (s) of
    the sample.

    PARAMETERS
    ----------
    dataSet : DataSet
        Instances whose continuous attribute values will be normalized.
    """
    def __init__(self, dataSet: DataSet):
        super().__init__(dataSet)
        averageInstance = dataSet.getInstanceList().average()
        standardDeviationInstance = dataSet.getInstanceList().standardDeviation()
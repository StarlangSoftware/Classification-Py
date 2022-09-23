from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.DataSet.DataSet import DataSet
from Classification.Filter.FeatureFilter import FeatureFilter
from Classification.Instance.Instance import Instance


class Normalize(FeatureFilter):

    __average_instance: Instance
    __standard_deviation_instance: Instance

    def __init__(self, dataSet: DataSet):
        """
        Constructor for normalize feature filter. It calculates and stores the mean (m) and standard deviation (s) of
        the sample.

        PARAMETERS
        ----------
        dataSet : DataSet
            Instances whose continuous attribute values will be normalized.
        """
        super().__init__(dataSet)
        self.__average_instance = dataSet.getInstanceList().average()
        self.__standard_deviation_instance = dataSet.getInstanceList().standardDeviation()

    def convertInstance(self, instance: Instance):
        """
        Normalizes the continuous attributes of a single instance. For all i, new x_i = (x_i - m_i) / s_i.

        PARAMETERS
        ----------
        instance : Instance
            Instance whose attributes will be normalized.
        """
        for i in range(instance.attributeSize()):
            if isinstance(instance.getAttribute(i), ContinuousAttribute):
                xi = instance.getAttribute(i)
                mi = self.__average_instance.getAttribute(i)
                si = self.__standard_deviation_instance.getAttribute(i)
                if isinstance(xi, ContinuousAttribute):
                    xi.setValue((xi.getValue() - mi.getValue()) / si.getValue())

    def convertDataDefinition(self):
        pass

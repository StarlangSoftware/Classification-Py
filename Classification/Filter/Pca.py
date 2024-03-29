from Math.Vector import Vector

from Classification.Attribute.AttributeType import AttributeType
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.DataSet.DataSet import DataSet
from Classification.Filter.TrainedFeatureFilter import TrainedFeatureFilter
from Classification.Instance.Instance import Instance


class Pca(TrainedFeatureFilter):

    __covariance_explained: float
    __eigenvectors: list
    __number_of_dimensions: int

    def __init__(self, dataSet: DataSet, covarianceExplained=0.99, numberOfDimensions=-1):
        """
        Constructor that sets the dataSet and covariance explained. Then calls train method.

        PARAMETERS
        ----------
        dataSet : DataSet
            DataSet that will bu used.
        covarianceExplained : float
            Number that shows the explained covariance.
        numberOfDimensions : int
            Dimension number.
        """
        super().__init__(dataSet)
        self.__eigenvectors = []
        self.__covariance_explained = covarianceExplained
        self.__number_of_dimensions = numberOfDimensions
        self.train()

    def __removeUnnecessaryEigenvectors(self):
        """
        The removeUnnecessaryEigenvectors methods takes an ArrayList of Eigenvectors. It first calculates the summation
        of eigenValues. Then it finds the eigenvectors which have lesser summation than covarianceExplained and removes
        these eigenvectors.
        """
        total = 0.0
        current_sum = 0.0
        for eigenvector in self.__eigenvectors:
            total += eigenvector.getEigenvalue()
        for i in range(len(self.__eigenvectors)):
            if current_sum / total < self.__covariance_explained:
                current_sum += self.__eigenvectors[i].getEigenvalue()
            else:
                del self.__eigenvectors[i:]
                break

    def __removeAllEigenvectorsExceptTheMostImportantK(self):
        """
        The removeAllEigenvectorsExceptTheMostImportantK method takes an list of Eigenvectors and removes the
        surplus eigenvectors when the number of eigenvectors is greater than the dimension.
        """
        del self.__eigenvectors[self.__number_of_dimensions:]

    def train(self):
        """
        The train method creates an averageVector from continuousAttributeAverage and a covariance {@link Matrix} from
        that averageVector. Then finds the eigenvectors of that covariance matrix and removes its unnecessary
        eigenvectors.
        """
        average_vector = Vector(self.dataSet.getInstanceList().continuousAverage())
        covariance = self.dataSet.getInstanceList().covariance(average_vector)
        self.__eigenvectors = covariance.characteristics()
        if self.__number_of_dimensions != -1:
            self.__removeAllEigenvectorsExceptTheMostImportantK()
        else:
            self.__removeUnnecessaryEigenvectors()

    def convertInstance(self, instance: Instance):
        """
        The convertInstance method takes an Instance as an input and creates a Vector attributes from continuous
        Attributes. After removing all attributes of given instance, it then adds new ContinuousAttribute by using the
        dot product of attributes Vector and the eigenvectors.

        PARAMETERS
        ----------
        instance : Instance
            Instance that will be converted to ContinuousAttribute by using eigenvectors.
        """
        attributes = Vector(instance.continuousAttributes())
        instance.removeAllAttributes()
        for eigenvector in self.__eigenvectors:
            instance.addAttribute(ContinuousAttribute(attributes.dotProduct(eigenvector)))

    def convertDataDefinition(self):
        """
        The convertDataDefinition method gets the data definitions of the dataSet and removes all the attributes. Then
        adds new attributes as CONTINUOUS.
        """
        data_definition = self.dataSet.getDataDefinition()
        data_definition.removeAllAtrributes()
        for i in range(len(self.__eigenvectors)):
            data_definition.addAttribute(AttributeType.CONTINUOUS)

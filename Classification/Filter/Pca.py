from Math.Vector import Vector

from Classification.Attribute.AttributeType import AttributeType
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.DataSet.DataSet import DataSet
from Classification.Filter.TrainedFeatureFilter import TrainedFeatureFilter
from Classification.Instance.Instance import Instance


class Pca(TrainedFeatureFilter):

    __covarianceExplained: float
    __eigenvectors: list
    __numberOfDimensions: int

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
    def __init__(self, dataSet: DataSet, covarianceExplained=0.99, numberOfDimensions=-1):
        super().__init__(dataSet)
        self.__eigenvectors = []
        self.__covarianceExplained = covarianceExplained
        if numberOfDimensions != -1:
            self.__numberOfDimensions = numberOfDimensions
        self.train()

    """
    The removeUnnecessaryEigenvectors methods takes an ArrayList of Eigenvectors. It first calculates the summation
    of eigenValues. Then it finds the eigenvectors which have lesser summation than covarianceExplained and removes 
    these eigenvectors.
    """
    def removeUnnecessaryEigenvectors(self):
        sum = 0.0
        currentSum = 0.0
        for eigenvector in self.__eigenvectors:
            sum += eigenvector.eigenvalue()
        for i in range(len(self.__eigenvectors)):
            if currentSum / sum < self.__covarianceExplained:
                currentSum += self.__eigenvectors[i].eigenValue()
            else:
                del self.__eigenvectors[i:]
                break

    """
    The removeAllEigenvectorsExceptTheMostImportantK method takes an list of Eigenvectors and removes the
    surplus eigenvectors when the number of eigenvectors is greater than the dimension.
    """
    def removeAllEigenvectorsExceptTheMostImportantK(self):
        del self.__eigenvectors[self.__numberOfDimensions:]

    """
    The train method creates an averageVector from continuousAttributeAverage and a covariance {@link Matrix} from that 
    averageVector. Then finds the eigenvectors of that covariance matrix and removes its unnecessary eigenvectors.
    """
    def train(self):
        averageVector = Vector(self.dataSet.getInstanceList().continuousAttributeAverage())
        covariance = self.dataSet.getInstanceList().covariance(averageVector)
        self.__eigenvectors = covariance.characteristics()
        if self.__numberOfDimensions != -1:
            self.removeAllEigenvectorsExceptTheMostImportantK()
        else:
            self.removeUnnecessaryEigenvectors()

    """
    The convertInstance method takes an {@link Instance} as an input and creates a Vector attributes from continuous
    Attributes. After removing all attributes of given instance, it then adds new ContinuousAttribute by using the dot
    product of attributes Vector and the eigenvectors.

    PARAMETERS
    ----------
    instance : Instance
        Instance that will be converted to ContinuousAttribute by using eigenvectors.
    """
    def convertInstance(self, instance: Instance):
        attributes = Vector(instance.continuousAttributes())
        instance.removeAllAttributes()
        for eigenvector in self.__eigenvectors:
            instance.addAttribute(ContinuousAttribute(attributes.dotProduct(eigenvector)))

    """
    The convertDataDefinition method gets the data definitions of the dataSet and removes all the attributes. Then adds
    new attributes as CONTINUOUS.
    """
    def convertDataDefinition(self):
        dataDefinition = self.dataSet.getDataDefinition()
        dataDefinition.removeAllAtrributes()
        for i in range(len(self.__eigenvectors)):
            dataDefinition.addAttribute(AttributeType.CONTINUOUS)

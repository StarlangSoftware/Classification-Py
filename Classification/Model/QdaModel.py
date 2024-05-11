from Math.DiscreteDistribution import DiscreteDistribution

from Classification.Instance.Instance import Instance
from Classification.Model.LdaModel import LdaModel


class QdaModel(LdaModel):

    __W: dict

    def constructor3(self,
                     priorDistribution: DiscreteDistribution,
                     W: dict,
                     w: dict,
                     w0: dict):
        """
        A constructor which sets the priorDistribution, w and w0 and dictionary of String Matrix according to given
        inputs.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            DiscreteDistribution input.
        W :
            Dict of String and Matrix.
        w : dict
            Dict of String and Vectors.
        w0 : dict
            Dict of String and float.
        """
        self.prior_distribution = priorDistribution
        self.__W = W
        self.w = w
        self.w0 = w0

    def constructor2(self, fileName: str):
        """
        Loads a quadratic discriminant analysis model from an input model file.
        :param fileName: Model file name.
        """
        inputFile = open(fileName, mode='r', encoding='utf-8')
        size = self.loadPriorDistribution(inputFile)
        self.loadWandW0(inputFile, size)
        self.__W = dict()
        for i in range(size):
            c = inputFile.readline().strip()
            matrix = self.loadMatrix(inputFile)
            self.__W[c] = matrix
        inputFile.close()

    def __init__(self,
                 priorDistribution: object,
                 W: dict = None,
                 w: dict = None,
                 w0: dict = None):
        super().__init__()
        if isinstance(priorDistribution, DiscreteDistribution):
            self.constructor3(priorDistribution, W, w, w0)
        elif isinstance(priorDistribution, str):
            self.constructor2(priorDistribution)

    def calculateMetric(self,
                        instance: Instance,
                        Ci: str) -> float:
        """
        The calculateMetric method takes an Instance and a String as inputs. It multiplies Matrix Wi with Vector xi
        then calculates the dot product of it with xi. Then, again it finds the dot product of wi and xi and returns the
        summation with w0i.

        PARAMETERS
        ----------
        instance : Instance
            Instance input.
        Ci : str
            String input.

        RETURNS
        -------
        float
            The result of Wi.multiplyWithVectorFromLeft(xi).dotProduct(xi) + wi.dotProduct(xi) + w0i.
        """
        xi = instance.toVector()
        Wi = self.__W[Ci]
        wi = self.w[Ci]
        w0i = self.w0[Ci]
        return Wi.multiplyWithVectorFromLeft(xi).dotProduct(xi) + wi.dotProduct(xi) + w0i

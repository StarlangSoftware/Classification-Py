from io import TextIOWrapper

from Math.DiscreteDistribution import DiscreteDistribution

from Classification.Instance.Instance import Instance
from Classification.Model.GaussianModel import GaussianModel


class LdaModel(GaussianModel):
    w0: dict
    w: dict

    def constructor1(self,
                     priorDistribution: DiscreteDistribution,
                     w: dict,
                     w0: dict):
        """
        A constructor which sets the priorDistribution, w and w0 according to given inputs.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            DiscreteDistribution input.
        w : dict
            Dict of String and Vectors.
        w0 : dict
            Dict of String and float.
        """
        self.prior_distribution = priorDistribution
        self.w = w
        self.w0 = w0

    def constructor2(self, fileName: str):
        inputFile = open(fileName, mode='r', encoding='utf-8')
        size = self.loadPriorDistribution(inputFile)
        self.loadWandW0(inputFile, size)
        inputFile.close()

    def __init__(self,
                 priorDistribution: object = None,
                 w: dict = None,
                 w0: dict = None):
        if priorDistribution is not None:
            if isinstance(priorDistribution, DiscreteDistribution):
                self.constructor1(priorDistribution, w, w0)
            elif isinstance(priorDistribution, str):
                self.constructor2(priorDistribution)

    def loadWandW0(self,
                   inputFile: TextIOWrapper,
                   size: int):
        self.w0 = dict()
        for i in range(size):
            line = inputFile.readline().strip()
            items = line.split(" ")
            self.w0[items[0]] = float(items[1])
        self.w = self.loadVectors(inputFile, size)

    def calculateMetric(self,
                        instance: Instance,
                        Ci: str) -> float:
        """
        The calculateMetric method takes an Instance and a String as inputs. It returns the dot product of given
        Instance and wi plus w0i.

        PARAMETERS
        ----------
        instance : Instance
            Instance input.
        Ci : str
            String input.

        RETURNS
        -------
        float
            The dot product of given Instance and wi plus w0i.
        """
        xi = instance.toVector()
        wi = self.w[Ci]
        w0i = self.w0[Ci]
        return wi.dotProduct(xi) + w0i

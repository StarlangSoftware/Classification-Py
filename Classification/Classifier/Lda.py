from Math.Matrix import Matrix
from Math.Vector import Vector

from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.LdaModel import LdaModel
from Classification.Parameter.Parameter import Parameter

import math


class Lda(Classifier):

    def train(self, trainSet: InstanceList, parameters: Parameter):
        """
        Training algorithm for the linear discriminant analysis classifier (Introduction to Machine Learning, Alpaydin,
        2015).

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        parameters : Parameter
            Parameter of the Lda algorithm.
        """
        w0 = {}
        w = {}
        priorDistribution = trainSet.classDistribution()
        classLists = Partition(trainSet)
        covariance = Matrix(trainSet.get(0).continuousAttributeSize(), trainSet.get(0).continuousAttributeSize())
        for i in range(classLists.size()):
            averageVector = Vector(classLists.get(i).continuousAverage())
            classCovariance = classLists.get(i).covariance(averageVector)
            classCovariance.multiplyWithConstant(classLists.get(i).size() - 1)
            covariance.add(classCovariance)
        covariance.divideByConstant(trainSet.size() - classLists.size())
        covariance.inverse()
        for i in range(classLists.size()):
            Ci = classLists.get(i).getClassLabel()
            averageVector = Vector(classLists.get(i).continuousAverage())
            wi = covariance.multiplyWithVectorFromRight(averageVector)
            w[Ci] = wi
            w0i = -0.5 * wi.dotProduct(averageVector) + math.log(priorDistribution.getProbability(Ci))
            w0[Ci] = w0i
        self.model = LdaModel(priorDistribution, w, w0)

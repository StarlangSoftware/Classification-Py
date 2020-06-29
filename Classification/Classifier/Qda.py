from Math.Vector import Vector
from copy import deepcopy

from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.QdaModel import QdaModel
from Classification.Parameter.Parameter import Parameter

import math


class Qda(Classifier):

    def train(self, trainSet: InstanceList, parameters: Parameter):
        """
        Training algorithm for the quadratic discriminant analysis classifier (Introduction to Machine Learning,
        Alpaydin, 2015).

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        """
        w0 = {}
        w = {}
        W = {}
        classLists = Partition(trainSet)
        priorDistribution = trainSet.classDistribution()
        for i in range(classLists.size()):
            Ci = classLists.get(i).getClassLabel()
            averageVector = Vector(classLists.get(i).continuousAverage())
            classCovariance = classLists.get(i).covariance(averageVector)
            determinant = classCovariance.determinant()
            classCovariance.inverse()
            Wi = deepcopy(classCovariance)
            Wi.multiplyWithConstant(-0.5)
            W[Ci] = Wi
            wi = classCovariance.multiplyWithVectorFromLeft(averageVector)
            w[Ci] = wi
            w0i = -0.5 * (wi.dotProduct(averageVector) + math.log(determinant)) + math.log(priorDistribution.
                                                                                           getProbability(Ci))
            w0[Ci] = w0i
        self.model = QdaModel(priorDistribution, W, w, w0)

from Math.Vector import Vector
from copy import deepcopy

from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.QdaModel import QdaModel
from Classification.Parameter.Parameter import Parameter

import math


class Qda(Classifier):

    def train(self,
              trainSet: InstanceList,
              parameters: Parameter):
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
        class_lists = Partition(trainSet)
        prior_distribution = trainSet.classDistribution()
        for i in range(class_lists.size()):
            Ci = class_lists.get(i).getClassLabel()
            average_vector = Vector(class_lists.get(i).continuousAverage())
            class_covariance = class_lists.get(i).covariance(average_vector)
            determinant = class_covariance.determinant()
            class_covariance.inverse()
            Wi = deepcopy(class_covariance)
            Wi.multiplyWithConstant(-0.5)
            W[Ci] = Wi
            wi = class_covariance.multiplyWithVectorFromLeft(average_vector)
            w[Ci] = wi
            w0i = -0.5 * (wi.dotProduct(average_vector) + math.log(determinant)) + math.log(prior_distribution.
                                                                                           getProbability(Ci))
            w0[Ci] = w0i
        self.model = QdaModel(prior_distribution, W, w, w0)

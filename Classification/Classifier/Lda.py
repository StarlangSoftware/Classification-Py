from Math.Matrix import Matrix
from Math.Vector import Vector

from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.LdaModel import LdaModel
from Classification.Parameter.Parameter import Parameter

import math


class Lda(Classifier):

    def train(self,
              trainSet: InstanceList,
              parameters: Parameter):
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
        prior_distribution = trainSet.classDistribution()
        class_lists = Partition(trainSet)
        covariance = Matrix(trainSet.get(0).continuousAttributeSize(), trainSet.get(0).continuousAttributeSize())
        for i in range(class_lists.size()):
            average_vector = Vector(class_lists.get(i).continuousAverage())
            class_covariance = class_lists.get(i).covariance(average_vector)
            class_covariance.multiplyWithConstant(class_lists.get(i).size() - 1)
            covariance.add(class_covariance)
        covariance.divideByConstant(trainSet.size() - class_lists.size())
        covariance.inverse()
        for i in range(class_lists.size()):
            Ci = class_lists.get(i).getClassLabel()
            average_vector = Vector(class_lists.get(i).continuousAverage())
            wi = covariance.multiplyWithVectorFromRight(average_vector)
            w[Ci] = wi
            w0i = -0.5 * wi.dotProduct(average_vector) + math.log(prior_distribution.getProbability(Ci))
            w0[Ci] = w0i
        self.model = LdaModel(prior_distribution, w, w0)

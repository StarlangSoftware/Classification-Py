import unittest

from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Filter.Pca import Pca
from Classification.Model.KnnModel import KnnModel
from Classification.Model.LinearPerceptronModel import LinearPerceptronModel
from Classification.Parameter.KnnParameter import KnnParameter
from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter
from test.Classifier.ClassifierTest import ClassifierTest


class PcaTest(ClassifierTest):

    def test_LinearPerceptron(self):
        linearPerceptron = LinearPerceptronModel()
        linearPerceptronParameter = LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100)
        pca = Pca(self.iris)
        pca.convert()
        linearPerceptron.train(self.iris.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(1.33, 100 * linearPerceptron.test(self.iris.getInstanceList()).getErrorRate(), 2)
        linearPerceptronParameter = LinearPerceptronParameter(1, 0.01, 0.99, 0.2, 100)
        pca = Pca(self.bupa)
        pca.convert()
        linearPerceptron.train(self.bupa.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(27.54, 100 * linearPerceptron.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        pca = Pca(self.dermatology)
        pca.convert()
        linearPerceptron.train(self.dermatology.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(3.28, 100 * linearPerceptron.test(self.dermatology.getInstanceList()).getErrorRate(), 2)

    def test_Knn(self):
        knn = KnnModel()
        knnParameter = KnnParameter(1, 3, EuclidianDistance())
        pca = Pca(self.iris)
        pca.convert()
        knn.train(self.iris.getInstanceList(), knnParameter)
        self.assertAlmostEqual(3.33, 100 * knn.test(self.iris.getInstanceList()).getErrorRate(), 2)
        pca = Pca(self.bupa)
        pca.convert()
        knn.train(self.bupa.getInstanceList(), knnParameter)
        self.assertAlmostEqual(19.13, 100 * knn.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        pca = Pca(self.dermatology)
        pca.convert()
        knn.train(self.dermatology.getInstanceList(), knnParameter)
        self.assertAlmostEqual(2.73, 100 * knn.test(self.dermatology.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

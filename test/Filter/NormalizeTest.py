import unittest

from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Filter.Normalize import Normalize
from Classification.Model.KnnModel import KnnModel
from Classification.Model.LinearPerceptronModel import LinearPerceptronModel
from Classification.Model.MultiLayerPerceptronModel import MultiLayerPerceptronModel
from Classification.Parameter.ActivationFunction import ActivationFunction
from Classification.Parameter.KnnParameter import KnnParameter
from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter
from Classification.Parameter.MultiLayerPerceptronParameter import MultiLayerPerceptronParameter
from test.Classifier.ClassifierTest import ClassifierTest


class NormalizeTest(ClassifierTest):

    def test_LinearPerceptron(self):
        linearPerceptron = LinearPerceptronModel()
        linearPerceptronParameter = LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100)
        normalize = Normalize(self.iris)
        normalize.convert()
        linearPerceptron.train(self.iris.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(2.00, 100 * linearPerceptron.test(self.iris.getInstanceList()).getErrorRate(), 2)
        normalize = Normalize(self.bupa)
        normalize.convert()
        linearPerceptron.train(self.bupa.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(26.67, 100 * linearPerceptron.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        normalize = Normalize(self.dermatology)
        normalize.convert()
        linearPerceptron.train(self.dermatology.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(1.91, 100 * linearPerceptron.test(self.dermatology.getInstanceList()).getErrorRate(), 2)

    def test_MultiLayerPerceptron(self):
        multiLayerPerceptron = MultiLayerPerceptronModel()
        multiLayerPerceptronParameter = MultiLayerPerceptronParameter(1, 1, 0.99, 0.2, 100, 3, ActivationFunction.SIGMOID)
        normalize = Normalize(self.iris)
        normalize.convert()
        multiLayerPerceptron.train(self.iris.getInstanceList(), multiLayerPerceptronParameter)
        self.assertAlmostEqual(1.33, 100 * multiLayerPerceptron.test(self.iris.getInstanceList()).getErrorRate(), 2)
        multiLayerPerceptronParameter = MultiLayerPerceptronParameter(1, 0.5, 0.99, 0.2, 100, 30, ActivationFunction.SIGMOID)
        normalize = Normalize(self.bupa)
        normalize.convert()
        multiLayerPerceptron.train(self.bupa.getInstanceList(), multiLayerPerceptronParameter)
        self.assertAlmostEqual(29.57, 100 * multiLayerPerceptron.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        multiLayerPerceptronParameter = MultiLayerPerceptronParameter(1, 0.1, 0.99, 0.2, 100, 20, ActivationFunction.SIGMOID)
        normalize = Normalize(self.dermatology)
        normalize.convert()
        multiLayerPerceptron.train(self.dermatology.getInstanceList(), multiLayerPerceptronParameter)
        self.assertAlmostEqual(3.28, 100 * multiLayerPerceptron.test(self.dermatology.getInstanceList()).getErrorRate(), 2)

    def test_Knn(self):
        knn = KnnModel()
        knnParameter = KnnParameter(1, 3, EuclidianDistance())
        normalize = Normalize(self.iris)
        normalize.convert()
        knn.train(self.iris.getInstanceList(), knnParameter)
        self.assertAlmostEqual(4.67, 100 * knn.test(self.iris.getInstanceList()).getErrorRate(), 2)
        normalize = Normalize(self.bupa)
        normalize.convert()
        knn.train(self.bupa.getInstanceList(), knnParameter)
        self.assertAlmostEqual(16.52, 100 * knn.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        normalize = Normalize(self.dermatology)
        normalize.convert()
        knn.train(self.dermatology.getInstanceList(), knnParameter)
        self.assertAlmostEqual(1.91, 100 * knn.test(self.dermatology.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

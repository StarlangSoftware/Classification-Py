import unittest

from Classification.Classifier.MultiLayerPerceptron import MultiLayerPerceptron
from Classification.Parameter.MultiLayerPerceptronParameter import MultiLayerPerceptronParameter
from test.Classifier.ClassifierTest import ClassifierTest


class MultiLayerPerceptronTest(ClassifierTest):

    def test_Train(self):
        multiLayerPerceptron = MultiLayerPerceptron()
        multiLayerPerceptronParameter = MultiLayerPerceptronParameter(1, 0.1, 0.99, 0.2, 100, 3)
        multiLayerPerceptron.train(self.iris.getInstanceList(), multiLayerPerceptronParameter)
        self.assertAlmostEqual(2.67, 100 * multiLayerPerceptron.test(self.iris.getInstanceList()).getErrorRate(), 2)
        multiLayerPerceptronParameter = MultiLayerPerceptronParameter(1, 0.01, 0.99, 0.2, 100, 30)
        multiLayerPerceptron.train(self.bupa.getInstanceList(), multiLayerPerceptronParameter)
        self.assertAlmostEqual(29.86, 100 * multiLayerPerceptron.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        multiLayerPerceptronParameter = MultiLayerPerceptronParameter(1, 0.01, 0.99, 0.2, 100, 20)
        multiLayerPerceptron.train(self.dermatology.getInstanceList(), multiLayerPerceptronParameter)
        self.assertAlmostEqual(3.55, 100 * multiLayerPerceptron.test(self.dermatology.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

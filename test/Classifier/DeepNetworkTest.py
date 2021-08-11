import unittest

from Classification.Classifier.DeepNetwork import DeepNetwork
from Classification.Parameter.ActivationFunction import ActivationFunction
from Classification.Parameter.DeepNetworkParameter import DeepNetworkParameter
from test.Classifier.ClassifierTest import ClassifierTest


class DeepNetworkTest(ClassifierTest):

    def test_Train(self):
        deepNetwork = DeepNetwork()
        deepNetworkParameter = DeepNetworkParameter(1, 0.1, 0.99, 0.2, 100, [5, 5], ActivationFunction.SIGMOID)
        deepNetwork.train(self.iris.getInstanceList(), deepNetworkParameter)
        self.assertAlmostEqual(4.00, 100 * deepNetwork.test(self.iris.getInstanceList()).getErrorRate(), 2)
        deepNetworkParameter = DeepNetworkParameter(1, 0.01, 0.99, 0.2, 100, [15, 15], ActivationFunction.SIGMOID)
        deepNetwork.train(self.bupa.getInstanceList(), deepNetworkParameter)
        self.assertAlmostEqual(28.12, 100 * deepNetwork.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        deepNetworkParameter = DeepNetworkParameter(1, 0.01, 0.99, 0.2, 100, [20], ActivationFunction.SIGMOID)
        deepNetwork.train(self.dermatology.getInstanceList(), deepNetworkParameter)
        self.assertAlmostEqual(3.55, 100 * deepNetwork.test(self.dermatology.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

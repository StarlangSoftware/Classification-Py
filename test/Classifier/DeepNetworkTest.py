import unittest

from Classification.Model.DeepNetworkModel import DeepNetworkModel
from Classification.Parameter.ActivationFunction import ActivationFunction
from Classification.Parameter.DeepNetworkParameter import DeepNetworkParameter
from test.Classifier.ClassifierTest import ClassifierTest


class DeepNetworkTest(ClassifierTest):

    def test_Train(self):
        deepNetwork = DeepNetworkModel()
        deepNetworkParameter = DeepNetworkParameter(1, 0.1, 0.99, 0.2, 100, [5, 5], ActivationFunction.SIGMOID)
        deepNetwork.train(self.iris.getInstanceList(), deepNetworkParameter)
        self.assertAlmostEqual(4.00, 100 * deepNetwork.test(self.iris.getInstanceList()).getErrorRate(), 2)
        deepNetworkParameter = DeepNetworkParameter(1, 0.01, 0.99, 0.2, 100, [15, 15], ActivationFunction.SIGMOID)
        deepNetwork.train(self.bupa.getInstanceList(), deepNetworkParameter)
        self.assertAlmostEqual(28.12, 100 * deepNetwork.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        deepNetworkParameter = DeepNetworkParameter(1, 0.01, 0.99, 0.2, 100, [20], ActivationFunction.SIGMOID)
        deepNetwork.train(self.dermatology.getInstanceList(), deepNetworkParameter)
        self.assertAlmostEqual(3.55, 100 * deepNetwork.test(self.dermatology.getInstanceList()).getErrorRate(), 2)

    def test_Load(self):
        deepNetwork = DeepNetworkModel()
        deepNetwork.loadModel("../../models/deepNetwork-iris.txt")
        self.assertAlmostEqual(1.33, 100 * deepNetwork.test(self.iris.getInstanceList()).getErrorRate(), 2)
        deepNetwork.loadModel("../../models/deepNetwork-bupa.txt")
        self.assertAlmostEqual(28.99, 100 * deepNetwork.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        deepNetwork.loadModel("../../models/deepNetwork-dermatology.txt")
        self.assertAlmostEqual(1.09, 100 * deepNetwork.test(self.dermatology.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

import unittest

from Classification.Model.NeuralNetwork.LinearPerceptronModel import LinearPerceptronModel
from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter
from test.Classifier.ClassifierTest import ClassifierTest


class LinearPerceptronTest(ClassifierTest):

    def test_Train(self):
        linearPerceptron = LinearPerceptronModel()
        linearPerceptronParameter = LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100)
        linearPerceptron.train(self.iris.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(1.33, 100 * linearPerceptron.test(self.iris.getInstanceList()).getErrorRate(), 2)
        linearPerceptron.train(self.bupa.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(28.99, 100 * linearPerceptron.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        linearPerceptron.train(self.dermatology.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(4.37, 100 * linearPerceptron.test(self.dermatology.getInstanceList()).getErrorRate(), 2)

    def test_Load(self):
        linearPerceptron = LinearPerceptronModel()
        linearPerceptron.loadModel("../../models/linearPerceptron-iris.txt")
        self.assertAlmostEqual(3.33, 100 * linearPerceptron.test(self.iris.getInstanceList()).getErrorRate(), 2)
        linearPerceptron.loadModel("../../models/linearPerceptron-bupa.txt")
        self.assertAlmostEqual(31.88, 100 * linearPerceptron.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        linearPerceptron.loadModel("../../models/linearPerceptron-dermatology.txt")
        self.assertAlmostEqual(0.82, 100 * linearPerceptron.test(self.dermatology.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

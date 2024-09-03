import unittest

from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Filter.LaryToBinary import LaryToBinary
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Model.KnnModel import KnnModel
from Classification.Model.LinearPerceptronModel import LinearPerceptronModel
from Classification.Parameter.C45Parameter import C45Parameter
from Classification.Parameter.KnnParameter import KnnParameter
from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter
from test.Classifier.ClassifierTest import ClassifierTest


class LaryToBinaryTest(ClassifierTest):

    def test_LinearPerceptron(self):
        linearPerceptron = LinearPerceptronModel()
        linearPerceptronParameter = LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100)
        discreteToIndexed = LaryToBinary(self.car)
        discreteToIndexed.convert()
        linearPerceptron.train(self.car.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(29.98, 100 * linearPerceptron.test(self.car.getInstanceList()).getErrorRate(), 2)
        discreteToIndexed = LaryToBinary(self.tictactoe)
        discreteToIndexed.convert()
        linearPerceptron.train(self.tictactoe.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(34.66, 100 * linearPerceptron.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)

    def test_Knn(self):
        knn = KnnModel()
        knnParameter = KnnParameter(1, 3, EuclidianDistance())
        discreteToIndexed = LaryToBinary(self.car)
        discreteToIndexed.convert()
        knn.train(self.car.getInstanceList(), knnParameter)
        self.assertAlmostEqual(20.31, 100 * knn.test(self.car.getInstanceList()).getErrorRate(), 2)

    def test_C45(self):
        c45 = DecisionTree()
        c45Parameter = C45Parameter(1, True, 0.2)
        discreteToIndexed = LaryToBinary(self.car)
        discreteToIndexed.convert()
        c45.train(self.car.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(3.99, 100 * c45.test(self.car.getInstanceList()).getErrorRate(), 2)
        discreteToIndexed = LaryToBinary(self.tictactoe)
        discreteToIndexed.convert()
        c45.train(self.tictactoe.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(3.34, 100 * c45.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)
        discreteToIndexed = LaryToBinary(self.nursery)
        discreteToIndexed.convert()
        c45.train(self.nursery.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(0.84, 100 * c45.test(self.nursery.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

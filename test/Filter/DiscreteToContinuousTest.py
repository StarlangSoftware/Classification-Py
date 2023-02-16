import unittest

from Classification.Classifier.C45 import C45
from Classification.Classifier.Knn import Knn
from Classification.Classifier.LinearPerceptron import LinearPerceptron
from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Filter.DiscreteToContinuous import DiscreteToContinuous
from Classification.Parameter.C45Parameter import C45Parameter
from Classification.Parameter.KnnParameter import KnnParameter
from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter
from test.Classifier.ClassifierTest import ClassifierTest


class DiscreteToContinuousTest(ClassifierTest):

    def test_LinearPerceptron(self):
        linearPerceptron = LinearPerceptron()
        linearPerceptronParameter = LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100)
        discreteToContinuous = DiscreteToContinuous(self.car)
        discreteToContinuous.convert()
        linearPerceptron.train(self.car.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(5.73, 100 * linearPerceptron.test(self.car.getInstanceList()).getErrorRate(), 2)
        discreteToContinuous = DiscreteToContinuous(self.tictactoe)
        discreteToContinuous.convert()
        linearPerceptron.train(self.tictactoe.getInstanceList(), linearPerceptronParameter)
        self.assertAlmostEqual(2.51, 100 * linearPerceptron.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)

    def test_Knn(self):
        knn = Knn()
        knnParameter = KnnParameter(1, 3, EuclidianDistance())
        discreteToContinuous = DiscreteToContinuous(self.car)
        discreteToContinuous.convert()
        knn.train(self.car.getInstanceList(), knnParameter)
        self.assertAlmostEqual(20.31, 100 * knn.test(self.car.getInstanceList()).getErrorRate(), 2)

    def test_C45(self):
        c45 = C45()
        c45Parameter = C45Parameter(1, True, 0.2)
        discreteToContinuous = DiscreteToContinuous(self.car)
        discreteToContinuous.convert()
        c45.train(self.car.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(3.99, 100 * c45.test(self.car.getInstanceList()).getErrorRate(), 2)
        discreteToContinuous = DiscreteToContinuous(self.tictactoe)
        discreteToContinuous.convert()
        c45.train(self.tictactoe.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(3.34, 100 * c45.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

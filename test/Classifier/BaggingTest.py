import unittest

from Classification.Classifier.Bagging import Bagging
from Classification.Parameter.BaggingParameter import BaggingParameter
from test.Classifier.ClassifierTest import ClassifierTest


class BaggingTest(ClassifierTest):

    def test_Train(self):
        bagging = Bagging()
        baggingParameter = BaggingParameter(1, 100)
        bagging.train(self.iris.getInstanceList(), baggingParameter)
        self.assertAlmostEqual(2.0, 100 * bagging.test(self.iris.getInstanceList()).getErrorRate(), 2)
        bagging.train(self.bupa.getInstanceList(), baggingParameter)
        self.assertAlmostEqual(42.03, 100 * bagging.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        bagging.train(self.dermatology.getInstanceList(), baggingParameter)
        self.assertAlmostEqual(2.46, 100 * bagging.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        bagging.train(self.car.getInstanceList(), baggingParameter)
        self.assertAlmostEqual(0.0, 100 * bagging.test(self.car.getInstanceList()).getErrorRate(), 2)
        bagging.train(self.tictactoe.getInstanceList(), baggingParameter)
        self.assertAlmostEqual(0.0, 100 * bagging.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

import unittest

from Classification.Classifier.RandomClassifier import RandomClassifier
from Classification.Parameter.Parameter import Parameter
from test.Classifier.ClassifierTest import ClassifierTest


class RandomClassifierTest(ClassifierTest):

    def test_Train(self):
        randomClassifier = RandomClassifier()
        parameter = Parameter(1)
        randomClassifier.train(self.iris.getInstanceList(), parameter)
        self.assertAlmostEqual(61.33, 100 * randomClassifier.test(self.iris.getInstanceList()).getErrorRate(), 2)
        randomClassifier.train(self.bupa.getInstanceList(), parameter)
        self.assertAlmostEqual(42.61, 100 * randomClassifier.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        randomClassifier.train(self.dermatology.getInstanceList(), parameter)
        self.assertAlmostEqual(83.88, 100 * randomClassifier.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        randomClassifier.train(self.car.getInstanceList(), parameter)
        self.assertAlmostEqual(75.06, 100 * randomClassifier.test(self.car.getInstanceList()).getErrorRate(), 2)
        randomClassifier.train(self.tictactoe.getInstanceList(), parameter)
        self.assertAlmostEqual(49.16, 100 * randomClassifier.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)
        randomClassifier.train(self.nursery.getInstanceList(), parameter)
        self.assertAlmostEqual(79.74, 100 * randomClassifier.test(self.nursery.getInstanceList()).getErrorRate(), 2)
        randomClassifier.train(self.chess.getInstanceList(), parameter)
        self.assertAlmostEqual(94.40, 100 * randomClassifier.test(self.chess.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

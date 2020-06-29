import unittest

from Classification.Classifier.RandomForest import RandomForest
from Classification.Parameter.RandomForestParameter import RandomForestParameter
from test.Classifier.ClassifierTest import ClassifierTest


class RandomForestTest(ClassifierTest):

    def test_Train(self):
        randomForest = RandomForest()
        randomForestParameter = RandomForestParameter(1, 100, 35)
        randomForest.train(self.iris.getInstanceList(), randomForestParameter)
        self.assertAlmostEqual(2.0, 100 * randomForest.test(self.iris.getInstanceList()).getErrorRate(), 2)
        randomForest.train(self.bupa.getInstanceList(), randomForestParameter)
        self.assertAlmostEqual(42.03, 100 * randomForest.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        randomForest.train(self.dermatology.getInstanceList(), randomForestParameter)
        self.assertAlmostEqual(2.46, 100 * randomForest.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        randomForest.train(self.car.getInstanceList(), randomForestParameter)
        self.assertAlmostEqual(0.0, 100 * randomForest.test(self.car.getInstanceList()).getErrorRate(), 2)
        randomForest.train(self.tictactoe.getInstanceList(), randomForestParameter)
        self.assertAlmostEqual(0.0, 100 * randomForest.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

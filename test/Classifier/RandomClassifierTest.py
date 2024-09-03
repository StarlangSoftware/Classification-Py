import unittest

from Classification.Model.RandomModel import RandomModel
from Classification.Parameter.Parameter import Parameter
from test.Classifier.ClassifierTest import ClassifierTest


class RandomClassifierTest(ClassifierTest):

    def test_Train(self):
        randomClassifier = RandomModel()
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

    def test_Load(self):
        randomClassifier = RandomModel()
        randomClassifier.loadModel("../../models/random-iris.txt")
        self.assertAlmostEqual(61.33, 100 * randomClassifier.test(self.iris.getInstanceList()).getErrorRate(), 2)
        randomClassifier.loadModel("../../models/random-bupa.txt")
        self.assertAlmostEqual(42.61, 100 * randomClassifier.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        randomClassifier.loadModel("../../models/random-dermatology.txt")
        self.assertAlmostEqual(83.88, 100 * randomClassifier.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        randomClassifier.loadModel("../../models/random-car.txt")
        self.assertAlmostEqual(75.06, 100 * randomClassifier.test(self.car.getInstanceList()).getErrorRate(), 2)
        randomClassifier.loadModel("../../models/random-tictactoe.txt")
        self.assertAlmostEqual(49.16, 100 * randomClassifier.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)
        randomClassifier.loadModel("../../models/random-nursery.txt")
        self.assertAlmostEqual(79.74, 100 * randomClassifier.test(self.nursery.getInstanceList()).getErrorRate(), 2)
        randomClassifier.loadModel("../../models/random-chess.txt")
        self.assertAlmostEqual(94.40, 100 * randomClassifier.test(self.chess.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

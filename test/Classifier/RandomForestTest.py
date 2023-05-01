import unittest

from Classification.Classifier.RandomForest import RandomForest
from Classification.Parameter.RandomForestParameter import RandomForestParameter
from test.Classifier.ClassifierTest import ClassifierTest


class RandomForestTest(ClassifierTest):

    def test_Train(self):
        randomForest = RandomForest()
        randomForestParameter = RandomForestParameter(1, 100, 35)
        randomForest.train(self.iris.getInstanceList(), randomForestParameter)
        self.assertAlmostEqual(0.0, 100 * randomForest.test(self.iris.getInstanceList()).getErrorRate(), 2)
        randomForest.train(self.bupa.getInstanceList(), randomForestParameter)
        self.assertAlmostEqual(0.0, 100 * randomForest.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        randomForest.train(self.dermatology.getInstanceList(), randomForestParameter)
        self.assertAlmostEqual(0.0, 100 * randomForest.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        randomForest.train(self.car.getInstanceList(), randomForestParameter)
        self.assertAlmostEqual(0.0, 100 * randomForest.test(self.car.getInstanceList()).getErrorRate(), 2)
        randomForest.train(self.tictactoe.getInstanceList(), randomForestParameter)
        self.assertAlmostEqual(0.0, 100 * randomForest.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)

    def test_Load(self):
        randomForest = RandomForest()
        randomForest.loadModel("../../models/randomForest-iris.txt")
        self.assertAlmostEqual(0.0, 100 * randomForest.test(self.iris.getInstanceList()).getErrorRate(), 2)
        randomForest.loadModel("../../models/randomForest-bupa.txt")
        self.assertAlmostEqual(0.0, 100 * randomForest.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        randomForest.loadModel("../../models/randomForest-dermatology.txt")
        self.assertAlmostEqual(0.0, 100 * randomForest.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        randomForest.loadModel("../../models/randomForest-car.txt")
        self.assertAlmostEqual(0.0, 100 * randomForest.test(self.car.getInstanceList()).getErrorRate(), 2)
        randomForest.loadModel("../../models/randomForest-tictactoe.txt")
        self.assertAlmostEqual(0.0, 100 * randomForest.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

import unittest

from Classification.Model.BaggingModel import BaggingModel
from Classification.Parameter.BaggingParameter import BaggingParameter
from test.Classifier.ClassifierTest import ClassifierTest


class BaggingTest(ClassifierTest):

    def test_Train(self):
        bagging = BaggingModel()
        baggingParameter = BaggingParameter(1, 100)
        bagging.train(self.iris.getInstanceList(), baggingParameter)
        self.assertAlmostEqual(0.0, 100 * bagging.test(self.iris.getInstanceList()).getErrorRate(), 2)
        bagging.train(self.bupa.getInstanceList(), baggingParameter)
        self.assertAlmostEqual(0.0, 100 * bagging.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        bagging.train(self.dermatology.getInstanceList(), baggingParameter)
        self.assertAlmostEqual(0.0, 100 * bagging.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        bagging.train(self.car.getInstanceList(), baggingParameter)
        self.assertAlmostEqual(0.0, 100 * bagging.test(self.car.getInstanceList()).getErrorRate(), 2)
        bagging.train(self.tictactoe.getInstanceList(), baggingParameter)
        self.assertAlmostEqual(0.0, 100 * bagging.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)

    def test_Load(self):
        bagging = BaggingModel()
        bagging.loadModel("../../models/bagging-iris.txt")
        self.assertAlmostEqual(0.0, 100 * bagging.test(self.iris.getInstanceList()).getErrorRate(), 2)
        bagging.loadModel("../../models/bagging-bupa.txt")
        self.assertAlmostEqual(0.0, 100 * bagging.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        bagging.loadModel("../../models/bagging-dermatology.txt")
        self.assertAlmostEqual(0.0, 100 * bagging.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        bagging.loadModel("../../models/bagging-car.txt")
        self.assertAlmostEqual(0.0, 100 * bagging.test(self.car.getInstanceList()).getErrorRate(), 2)
        bagging.loadModel("../../models/bagging-tictactoe.txt")
        self.assertAlmostEqual(0.0, 100 * bagging.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

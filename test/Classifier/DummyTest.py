import unittest

from Classification.Classifier.Dummy import Dummy
from test.Classifier.ClassifierTest import ClassifierTest


class DummyTest(ClassifierTest):

    def test_Train(self):
        dummy = Dummy()
        dummy.train(self.iris.getInstanceList())
        self.assertAlmostEqual(66.67, 100 * dummy.test(self.iris.getInstanceList()).getErrorRate(), 2)
        dummy.train(self.bupa.getInstanceList())
        self.assertAlmostEqual(42.03, 100 * dummy.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        dummy.train(self.dermatology.getInstanceList())
        self.assertAlmostEqual(69.40, 100 * dummy.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        dummy.train(self.car.getInstanceList())
        self.assertAlmostEqual(29.98, 100 * dummy.test(self.car.getInstanceList()).getErrorRate(), 2)
        dummy.train(self.tictactoe.getInstanceList())
        self.assertAlmostEqual(34.66, 100 * dummy.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)
        dummy.train(self.nursery.getInstanceList())
        self.assertAlmostEqual(66.67, 100 * dummy.test(self.nursery.getInstanceList()).getErrorRate(), 2)
        dummy.train(self.chess.getInstanceList())
        self.assertAlmostEqual(83.77, 100 * dummy.test(self.chess.getInstanceList()).getErrorRate(), 2)

    def test_Load(self):
        dummy = Dummy()
        dummy.loadModel("../../models/dummy-iris.txt")
        self.assertAlmostEqual(66.67, 100 * dummy.test(self.iris.getInstanceList()).getErrorRate(), 2)
        dummy.loadModel("../../models/dummy-bupa.txt")
        self.assertAlmostEqual(42.03, 100 * dummy.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        dummy.loadModel("../../models/dummy-dermatology.txt")
        self.assertAlmostEqual(69.40, 100 * dummy.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        dummy.loadModel("../../models/dummy-car.txt")
        self.assertAlmostEqual(29.98, 100 * dummy.test(self.car.getInstanceList()).getErrorRate(), 2)
        dummy.loadModel("../../models/dummy-tictactoe.txt")
        self.assertAlmostEqual(34.66, 100 * dummy.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)
        dummy.loadModel("../../models/dummy-nursery.txt")
        self.assertAlmostEqual(66.67, 100 * dummy.test(self.nursery.getInstanceList()).getErrorRate(), 2)
        dummy.loadModel("../../models/dummy-chess.txt")
        self.assertAlmostEqual(83.77, 100 * dummy.test(self.chess.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

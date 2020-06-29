import unittest

from Classification.Classifier.Dummy import Dummy
from test.Classifier.ClassifierTest import ClassifierTest


class DummyTest(ClassifierTest):

    def test_Train(self):
        dummy = Dummy()
        dummy.train(self.iris.getInstanceList(), None)
        self.assertAlmostEqual(66.67, 100 * dummy.test(self.iris.getInstanceList()).getErrorRate(), 2)
        dummy.train(self.bupa.getInstanceList(), None)
        self.assertAlmostEqual(42.03, 100 * dummy.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        dummy.train(self.dermatology.getInstanceList(), None)
        self.assertAlmostEqual(69.40, 100 * dummy.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        dummy.train(self.car.getInstanceList(), None)
        self.assertAlmostEqual(29.98, 100 * dummy.test(self.car.getInstanceList()).getErrorRate(), 2)
        dummy.train(self.tictactoe.getInstanceList(), None)
        self.assertAlmostEqual(34.66, 100 * dummy.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)
        dummy.train(self.nursery.getInstanceList(), None)
        self.assertAlmostEqual(66.67, 100 * dummy.test(self.nursery.getInstanceList()).getErrorRate(), 2)
        dummy.train(self.chess.getInstanceList(), None)
        self.assertAlmostEqual(83.77, 100 * dummy.test(self.chess.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

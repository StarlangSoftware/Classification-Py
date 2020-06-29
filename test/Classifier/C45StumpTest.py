import unittest

from Classification.Classifier.C45Stump import C45Stump
from test.Classifier.ClassifierTest import ClassifierTest


class C45StumpTest(ClassifierTest):

    def test_Train(self):
        c45Stump = C45Stump()
        c45Stump.train(self.iris.getInstanceList(), None)
        self.assertAlmostEqual(33.33, 100 * c45Stump.test(self.iris.getInstanceList()).getErrorRate(), 2)
        c45Stump.train(self.bupa.getInstanceList(), None)
        self.assertAlmostEqual(42.03, 100 * c45Stump.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        c45Stump.train(self.dermatology.getInstanceList(), None)
        self.assertAlmostEqual(49.73, 100 * c45Stump.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        c45Stump.train(self.car.getInstanceList(), None)
        self.assertAlmostEqual(29.98, 100 * c45Stump.test(self.car.getInstanceList()).getErrorRate(), 2)
        c45Stump.train(self.tictactoe.getInstanceList(), None)
        self.assertAlmostEqual(30.06, 100 * c45Stump.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)
        c45Stump.train(self.nursery.getInstanceList(), None)
        self.assertAlmostEqual(29.03, 100 * c45Stump.test(self.nursery.getInstanceList()).getErrorRate(), 2)
        c45Stump.train(self.chess.getInstanceList(), None)
        self.assertAlmostEqual(80.76, 100 * c45Stump.test(self.chess.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

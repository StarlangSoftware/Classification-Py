import unittest

from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Parameter.C45Parameter import C45Parameter
from test.Classifier.ClassifierTest import ClassifierTest


class C45Test(ClassifierTest):

    def test_Train(self):
        c45 = DecisionTree()
        c45Parameter = C45Parameter(1, True, 0.2)
        c45.train(self.iris.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(2.00, 100 * c45.test(self.iris.getInstanceList()).getErrorRate(), 2)
        c45.train(self.bupa.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(30.43, 100 * c45.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        c45.train(self.dermatology.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(4.37, 100 * c45.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        c45.train(self.car.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(6.48, 100 * c45.test(self.car.getInstanceList()).getErrorRate(), 2)
        c45.train(self.tictactoe.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(8.77, 100 * c45.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)

    def test_Load(self):
        c45 = DecisionTree()
        c45.loadModel("../../models/c45-iris.txt")
        self.assertAlmostEqual(4.00, 100 * c45.test(self.iris.getInstanceList()).getErrorRate(), 2)
        c45.loadModel("../../models/c45-bupa.txt")
        self.assertAlmostEqual(42.03, 100 * c45.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        c45.loadModel("../../models/c45-dermatology.txt")
        self.assertAlmostEqual(2.19, 100 * c45.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        c45.loadModel("../../models/c45-car.txt")
        self.assertAlmostEqual(8.16, 100 * c45.test(self.car.getInstanceList()).getErrorRate(), 2)
        c45.loadModel("../../models/c45-tictactoe.txt")
        self.assertAlmostEqual(14.61, 100 * c45.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

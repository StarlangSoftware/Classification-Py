import unittest

from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Model.NonParametric.KnnModel import KnnModel
from Classification.Parameter.KnnParameter import KnnParameter
from test.Classifier.ClassifierTest import ClassifierTest


class KnnTest(ClassifierTest):

    def test_Train(self):
        knn = KnnModel()
        knnParameter = KnnParameter(1, 3, EuclidianDistance())
        knn.train(self.iris.getInstanceList(), knnParameter)
        self.assertAlmostEqual(4.00, 100 * knn.test(self.iris.getInstanceList()).getErrorRate(), 2)
        knn.train(self.bupa.getInstanceList(), knnParameter)
        self.assertAlmostEqual(19.42, 100 * knn.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        knn.train(self.dermatology.getInstanceList(), knnParameter)
        self.assertAlmostEqual(3.01, 100 * knn.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        knn.train(self.car.getInstanceList(), knnParameter)
        self.assertAlmostEqual(20.31, 100 * knn.test(self.car.getInstanceList()).getErrorRate(), 2)
        knn.train(self.tictactoe.getInstanceList(), knnParameter)
        self.assertAlmostEqual(32.57, 100 * knn.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)

    def test_Load(self):
        knn = KnnModel()
        knn.loadModel("../../models/knn-iris.txt")
        self.assertAlmostEqual(4.00, 100 * knn.test(self.iris.getInstanceList()).getErrorRate(), 2)
        knn.loadModel("../../models/knn-bupa.txt")
        self.assertAlmostEqual(19.42, 100 * knn.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        knn.loadModel("../../models/knn-dermatology.txt")
        self.assertAlmostEqual(3.01, 100 * knn.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        knn.loadModel("../../models/knn-car.txt")
        self.assertAlmostEqual(20.31, 100 * knn.test(self.car.getInstanceList()).getErrorRate(), 2)
        knn.loadModel("../../models/knn-tictactoe.txt")
        self.assertAlmostEqual(32.57, 100 * knn.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

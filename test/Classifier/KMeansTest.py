import unittest

from Classification.Model.KMeansModel import KMeansModel
from Classification.Parameter.KMeansParameter import KMeansParameter
from test.Classifier.ClassifierTest import ClassifierTest


class KMeansTest(ClassifierTest):

    def test_Train(self):
        kMeans = KMeansModel()
        kMeansParameter = KMeansParameter(1)
        kMeans.train(self.iris.getInstanceList(), kMeansParameter)
        self.assertAlmostEqual(7.33, 100 * kMeans.test(self.iris.getInstanceList()).getErrorRate(), 2)
        kMeans.train(self.bupa.getInstanceList(), kMeansParameter)
        self.assertAlmostEqual(43.77, 100 * kMeans.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        kMeans.train(self.dermatology.getInstanceList(), kMeansParameter)
        self.assertAlmostEqual(45.08, 100 * kMeans.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        kMeans.train(self.car.getInstanceList(), kMeansParameter)
        self.assertAlmostEqual(47.97, 100 * kMeans.test(self.car.getInstanceList()).getErrorRate(), 2)
        kMeans.train(self.tictactoe.getInstanceList(), kMeansParameter)
        self.assertAlmostEqual(38.94, 100 * kMeans.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)
        kMeans.train(self.nursery.getInstanceList(), kMeansParameter)
        self.assertAlmostEqual(53.60, 100 * kMeans.test(self.nursery.getInstanceList()).getErrorRate(), 2)
        kMeans.train(self.chess.getInstanceList(), kMeansParameter)
        self.assertAlmostEqual(83.25, 100 * kMeans.test(self.chess.getInstanceList()).getErrorRate(), 2)

    def test_Load(self):
        kMeans = KMeansModel()
        kMeans.loadModel("../../models/kMeans-iris.txt")
        self.assertAlmostEqual(7.33, 100 * kMeans.test(self.iris.getInstanceList()).getErrorRate(), 2)
        kMeans.loadModel("../../models/kMeans-bupa.txt")
        self.assertAlmostEqual(43.77, 100 * kMeans.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        kMeans.loadModel("../../models/kMeans-dermatology.txt")
        self.assertAlmostEqual(45.08, 100 * kMeans.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        kMeans.loadModel("../../models/kMeans-car.txt")
        self.assertAlmostEqual(44.21, 100 * kMeans.test(self.car.getInstanceList()).getErrorRate(), 2)
        kMeans.loadModel("../../models/kMeans-tictactoe.txt")
        self.assertAlmostEqual(38.94, 100 * kMeans.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)
        kMeans.loadModel("../../models/kMeans-nursery.txt")
        self.assertAlmostEqual(60.26, 100 * kMeans.test(self.nursery.getInstanceList()).getErrorRate(), 2)
        kMeans.loadModel("../../models/kMeans-chess.txt")
        self.assertAlmostEqual(83.25, 100 * kMeans.test(self.chess.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

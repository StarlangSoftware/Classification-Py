import unittest

from Classification.Model.NaiveBayesModel import NaiveBayesModel
from test.Classifier.ClassifierTest import ClassifierTest


class NaiveBayesTest(ClassifierTest):

    def test_Train(self):
        naiveBayes = NaiveBayesModel()
        naiveBayes.train(self.iris.getInstanceList())
        self.assertAlmostEqual(5.33, 100 * naiveBayes.test(self.iris.getInstanceList()).getErrorRate(), 2)
        naiveBayes.train(self.bupa.getInstanceList())
        self.assertAlmostEqual(38.55, 100 * naiveBayes.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        naiveBayes.train(self.dermatology.getInstanceList())
        self.assertAlmostEqual(9.56, 100 * naiveBayes.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        naiveBayes.train(self.car.getInstanceList())
        self.assertAlmostEqual(12.91, 100 * naiveBayes.test(self.car.getInstanceList()).getErrorRate(), 2)
        naiveBayes.train(self.tictactoe.getInstanceList())
        self.assertAlmostEqual(30.17, 100 * naiveBayes.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)
        naiveBayes.train(self.nursery.getInstanceList())
        self.assertAlmostEqual(9.70, 100 * naiveBayes.test(self.nursery.getInstanceList()).getErrorRate(), 2)

    def test_Load(self):
        naiveBayes = NaiveBayesModel()
        naiveBayes.loadModel("../../models/naiveBayes-iris.txt")
        self.assertAlmostEqual(5.33, 100 * naiveBayes.test(self.iris.getInstanceList()).getErrorRate(), 2)
        naiveBayes.loadModel("../../models/naiveBayes-bupa.txt")
        self.assertAlmostEqual(38.55, 100 * naiveBayes.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        naiveBayes.loadModel("../../models/naiveBayes-dermatology.txt")
        self.assertAlmostEqual(9.56, 100 * naiveBayes.test(self.dermatology.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

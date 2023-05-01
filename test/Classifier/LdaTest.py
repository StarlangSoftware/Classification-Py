import unittest

from Classification.Classifier.Lda import Lda
from test.Classifier.ClassifierTest import ClassifierTest


class LdaTest(ClassifierTest):

    def test_Train(self):
        lda = Lda()
        lda.train(self.iris.getInstanceList())
        self.assertAlmostEqual(2.00, 100 * lda.test(self.iris.getInstanceList()).getErrorRate(), 2)
        lda.train(self.bupa.getInstanceList())
        self.assertAlmostEqual(29.57, 100 * lda.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        lda.train(self.dermatology.getInstanceList())
        self.assertAlmostEqual(1.91, 100 * lda.test(self.dermatology.getInstanceList()).getErrorRate(), 2)

    def test_Load(self):
        lda = Lda()
        lda.loadModel("../../models/lda-iris.txt")
        self.assertAlmostEqual(2.00, 100 * lda.test(self.iris.getInstanceList()).getErrorRate(), 2)
        lda.loadModel("../../models/lda-bupa.txt")
        self.assertAlmostEqual(29.57, 100 * lda.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        lda.loadModel("../../models/lda-dermatology.txt")
        self.assertAlmostEqual(1.91, 100 * lda.test(self.dermatology.getInstanceList()).getErrorRate(), 2)

if __name__ == '__main__':
    unittest.main()

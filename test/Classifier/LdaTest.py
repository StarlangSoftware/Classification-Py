import unittest

from Classification.Classifier.Lda import Lda
from test.Classifier.ClassifierTest import ClassifierTest


class LdaTest(ClassifierTest):

    def test_Train(self):
        lda = Lda()
        lda.train(self.iris.getInstanceList(), None)
        self.assertAlmostEqual(2.00, 100 * lda.test(self.iris.getInstanceList()).getErrorRate(), 2)
        lda.train(self.bupa.getInstanceList(), None)
        self.assertAlmostEqual(29.57, 100 * lda.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        lda.train(self.dermatology.getInstanceList(), None)
        self.assertAlmostEqual(1.91, 100 * lda.test(self.dermatology.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

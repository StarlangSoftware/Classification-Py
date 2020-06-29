import unittest

from Classification.Classifier.Qda import Qda
from test.Classifier.ClassifierTest import ClassifierTest


class QdaTest(ClassifierTest):

    def test_Train(self):
        qda = Qda()
        qda.train(self.iris.getInstanceList(), None)
        self.assertAlmostEqual(2.00, 100 * qda.test(self.iris.getInstanceList()).getErrorRate(), 2)
        qda.train(self.bupa.getInstanceList(), None)
        self.assertAlmostEqual(36.52, 100 * qda.test(self.bupa.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

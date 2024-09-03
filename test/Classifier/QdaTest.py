import unittest

from Classification.Model.Parametric.QdaModel import QdaModel
from test.Classifier.ClassifierTest import ClassifierTest


class QdaTest(ClassifierTest):

    def test_Train(self):
        qda = QdaModel()
        qda.train(self.iris.getInstanceList())
        self.assertAlmostEqual(2.00, 100 * qda.test(self.iris.getInstanceList()).getErrorRate(), 2)
        qda.train(self.bupa.getInstanceList())
        self.assertAlmostEqual(36.52, 100 * qda.test(self.bupa.getInstanceList()).getErrorRate(), 2)

    def test_Load(self):
        qda = QdaModel()
        qda.loadModel("../../models/qda-iris.txt")
        self.assertAlmostEqual(2.00, 100 * qda.test(self.iris.getInstanceList()).getErrorRate(), 2)
        qda.loadModel("../../models/qda-bupa.txt")
        self.assertAlmostEqual(36.52, 100 * qda.test(self.bupa.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

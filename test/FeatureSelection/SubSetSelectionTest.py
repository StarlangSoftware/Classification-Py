import unittest

from Classification.Classifier.C45 import C45
from Classification.Classifier.Knn import Knn
from Classification.Classifier.Lda import Lda
from Classification.Classifier.NaiveBayes import NaiveBayes
from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRun import KFoldRun
from Classification.FeatureSelection.BackwardSelection import BackwardSelection
from Classification.FeatureSelection.FloatingSelection import FloatingSelection
from Classification.FeatureSelection.ForwardSelection import ForwardSelection
from Classification.Parameter.C45Parameter import C45Parameter
from Classification.Parameter.KnnParameter import KnnParameter
from Classification.Parameter.Parameter import Parameter
from test.Classifier.ClassifierTest import ClassifierTest


class SubSetSelectionTest(ClassifierTest):

    def test_SubSetSelectionC45(self):
        kFoldRun = KFoldRun(10)
        forwardSelection = ForwardSelection()
        experiment = Experiment(C45(), C45Parameter(1, True, 0.2), self.iris)
        self.assertAlmostEqual(1, forwardSelection.execute(kFoldRun, experiment).size())
        backwardSelection = BackwardSelection(self.iris.attributeCount())
        self.assertAlmostEqual(3, backwardSelection.execute(kFoldRun, experiment).size())
        floatingSelection = FloatingSelection()
        self.assertAlmostEqual(1, floatingSelection.execute(kFoldRun, experiment).size())

    def test_SubSetSelectionNaiveBayes(self):
        kFoldRun = KFoldRun(10)
        forwardSelection = ForwardSelection()
        experiment = Experiment(NaiveBayes(), Parameter(1), self.nursery)
        self.assertAlmostEqual(3, forwardSelection.execute(kFoldRun, experiment).size())
        backwardSelection = BackwardSelection(self.nursery.attributeCount())
        self.assertAlmostEqual(8, backwardSelection.execute(kFoldRun, experiment).size())
        floatingSelection = FloatingSelection()
        self.assertAlmostEqual(3, floatingSelection.execute(kFoldRun, experiment).size())

    def test_SubSetSelectionLda(self):
        kFoldRun = KFoldRun(10)
        forwardSelection = ForwardSelection()
        experiment = Experiment(Lda(), Parameter(1), self.dermatology)
        self.assertAlmostEqual(11, forwardSelection.execute(kFoldRun, experiment).size())
        backwardSelection = BackwardSelection(self.dermatology.attributeCount())
        self.assertAlmostEqual(33, backwardSelection.execute(kFoldRun, experiment).size())
        floatingSelection = FloatingSelection()
        self.assertAlmostEqual(11, floatingSelection.execute(kFoldRun, experiment).size())

    def test_SubSetSelectionKnn(self):
        kFoldRun = KFoldRun(10)
        forwardSelection = ForwardSelection()
        experiment = Experiment(Knn(), KnnParameter(1, 3), self.car)
        self.assertAlmostEqual(5, forwardSelection.execute(kFoldRun, experiment).size())
        backwardSelection = BackwardSelection(self.car.attributeCount())
        self.assertAlmostEqual(5, backwardSelection.execute(kFoldRun, experiment).size())
        floatingSelection = FloatingSelection()
        self.assertAlmostEqual(5, floatingSelection.execute(kFoldRun, experiment).size())


if __name__ == '__main__':
    unittest.main()

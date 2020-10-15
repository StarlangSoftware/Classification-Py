import unittest

from Classification.Classifier.Bagging import Bagging
from Classification.Classifier.C45 import C45
from Classification.Classifier.Dummy import Dummy
from Classification.Classifier.Knn import Knn
from Classification.Classifier.Lda import Lda
from Classification.Classifier.LinearPerceptron import LinearPerceptron
from Classification.Classifier.NaiveBayes import NaiveBayes
from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRun import KFoldRun
from Classification.Parameter.BaggingParameter import BaggingParameter
from Classification.Parameter.C45Parameter import C45Parameter
from Classification.Parameter.KnnParameter import KnnParameter
from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter
from Classification.Parameter.Parameter import Parameter
from test.Classifier.ClassifierTest import ClassifierTest


class KFoldRunTest(ClassifierTest):

    def test_Execute(self):
        kFoldRun = KFoldRun(10)
        experimentPerformance = kFoldRun.execute(Experiment(C45(), C45Parameter(1, True, 0.2), self.iris))
        self.assertAlmostEqual(6.00, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(C45(), C45Parameter(1, True, 0.2), self.tictactoe))
        self.assertAlmostEqual(18.78, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(Knn(), KnnParameter(1, 3, EuclidianDistance()), self.bupa))
        self.assertAlmostEqual(36.85, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(Knn(), KnnParameter(1, 3, EuclidianDistance()), self.dermatology))
        self.assertAlmostEqual(10.92, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(Lda(), Parameter(1), self.bupa))
        self.assertAlmostEqual(31.61, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(Lda(), Parameter(1), self.dermatology))
        self.assertAlmostEqual(3.30, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(LinearPerceptron(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.iris))
        self.assertAlmostEqual(5.33, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(LinearPerceptron(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.dermatology))
        self.assertAlmostEqual(3.81, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(NaiveBayes(), Parameter(1), self.car))
        self.assertAlmostEqual(14.88, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(NaiveBayes(), Parameter(1), self.nursery))
        self.assertAlmostEqual(9.71, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(Bagging(), BaggingParameter(1, 50), self.tictactoe))
        self.assertAlmostEqual(3.55, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(Bagging(), BaggingParameter(1, 50), self.car))
        self.assertAlmostEqual(6.77, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(Dummy(), Parameter(1), self.nursery))
        self.assertAlmostEqual(67.12, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = kFoldRun.execute(Experiment(Dummy(), Parameter(1), self.iris))
        self.assertAlmostEqual(79.33, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

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
from Classification.Experiment.MxKFoldRun import MxKFoldRun
from Classification.Parameter.BaggingParameter import BaggingParameter
from Classification.Parameter.C45Parameter import C45Parameter
from Classification.Parameter.KnnParameter import KnnParameter
from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter
from Classification.Parameter.Parameter import Parameter
from test.Classifier.ClassifierTest import ClassifierTest


class MxKFoldRunTest(ClassifierTest):

    def test_Execute(self):
        mxKFoldRun = MxKFoldRun(5, 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(C45(), C45Parameter(1, True, 0.2), self.iris))
        self.assertAlmostEqual(6.13, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(C45(), C45Parameter(1, True, 0.2), self.tictactoe))
        self.assertAlmostEqual(23.51, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(Knn(), KnnParameter(1, 3, EuclidianDistance()), self.bupa))
        self.assertAlmostEqual(37.05, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(Knn(), KnnParameter(1, 3, EuclidianDistance()), self.dermatology))
        self.assertAlmostEqual(15.41, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(Lda(), Parameter(1), self.bupa))
        self.assertAlmostEqual(34.72, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(Lda(), Parameter(1), self.dermatology))
        self.assertAlmostEqual(4.04, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(LinearPerceptron(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.iris))
        self.assertAlmostEqual(5.2, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(LinearPerceptron(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.dermatology))
        self.assertAlmostEqual(5.46, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(NaiveBayes(), Parameter(1), self.car))
        self.assertAlmostEqual(16.52, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(NaiveBayes(), Parameter(1), self.nursery))
        self.assertAlmostEqual(9.80, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(Bagging(), BaggingParameter(1, 50), self.tictactoe))
        self.assertAlmostEqual(8.77, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(Bagging(), BaggingParameter(1, 50), self.car))
        self.assertAlmostEqual(9.77, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(Dummy(), Parameter(1), self.nursery))
        self.assertAlmostEqual(67.09, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(Dummy(), Parameter(1), self.iris))
        self.assertAlmostEqual(70.53, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

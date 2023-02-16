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
from Classification.StatisticalTest.Pairedt import Pairedt
from test.Classifier.ClassifierTest import ClassifierTest


class PairedtTest(ClassifierTest):

    def test_Compare(self):
        kFoldRun = KFoldRun(10)
        pairedt = Pairedt()
        experimentPerformance1 = kFoldRun.execute(Experiment(C45(), C45Parameter(1, True, 0.2), self.iris))
        experimentPerformance2 = kFoldRun.execute(Experiment(LinearPerceptron(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.iris))
        self.assertAlmostEqual(0.278, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 3)
        experimentPerformance1 = kFoldRun.execute(Experiment(C45(), C45Parameter(1, True, 0.2), self.tictactoe))
        experimentPerformance2 = kFoldRun.execute(Experiment(Bagging(), BaggingParameter(1, 50), self.tictactoe))
        self.assertAlmostEqual(0.00000692, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 7)
        experimentPerformance1 = kFoldRun.execute(Experiment(Lda(), Parameter(1), self.dermatology))
        experimentPerformance2 = kFoldRun.execute(Experiment(LinearPerceptron(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.dermatology))
        self.assertAlmostEqual(0.7842, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 4)
        experimentPerformance1 = kFoldRun.execute(Experiment(Dummy(), Parameter(1), self.nursery))
        experimentPerformance2 = kFoldRun.execute(Experiment(NaiveBayes(), Parameter(1), self.nursery))
        self.assertAlmostEqual(0.0, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 4)
        experimentPerformance1 = kFoldRun.execute(Experiment(NaiveBayes(), Parameter(1), self.car))
        experimentPerformance2 = kFoldRun.execute(Experiment(Bagging(), BaggingParameter(1, 50), self.car))
        self.assertAlmostEqual(0.00000336, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 7)
        experimentPerformance1 = kFoldRun.execute(Experiment(Knn(), KnnParameter(1, 3, EuclidianDistance()), self.bupa))
        experimentPerformance2 = kFoldRun.execute(Experiment(Lda(), Parameter(1), self.bupa))
        self.assertAlmostEqual(0.1640, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 4)


if __name__ == '__main__':
    unittest.main()

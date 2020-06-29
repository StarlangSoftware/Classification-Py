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
from Classification.StatisticalTest.Combined5x2t import Combined5x2t
from test.Classifier.ClassifierTest import ClassifierTest

class Combined5x2tTest(ClassifierTest):

    def test_Compare(self):
        mxKFoldRun = MxKFoldRun(5, 2)
        combined5x2t = Combined5x2t()
        experimentPerformance1 = mxKFoldRun.execute(Experiment(C45(), C45Parameter(1, True, 0.2), self.iris))
        experimentPerformance2 = mxKFoldRun.execute(Experiment(LinearPerceptron(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.iris))
        self.assertAlmostEqual(0.157, combined5x2t.compare(experimentPerformance1, experimentPerformance2).getPValue(), 3)
        experimentPerformance1 = mxKFoldRun.execute(Experiment(C45(), C45Parameter(1, True, 0.2), self.tictactoe))
        experimentPerformance2 = mxKFoldRun.execute(Experiment(Bagging(), BaggingParameter(1, 50), self.tictactoe))
        self.assertAlmostEqual(0.00044, combined5x2t.compare(experimentPerformance1, experimentPerformance2).getPValue(), 5)
        experimentPerformance1 = mxKFoldRun.execute(Experiment(Lda(), Parameter(1), self.dermatology))
        experimentPerformance2 = mxKFoldRun.execute(Experiment(LinearPerceptron(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.dermatology))
        self.assertAlmostEqual(0.9819, combined5x2t.compare(experimentPerformance1, experimentPerformance2).getPValue(), 4)
        experimentPerformance1 = mxKFoldRun.execute(Experiment(Dummy(), Parameter(1), self.nursery))
        experimentPerformance2 = mxKFoldRun.execute(Experiment(NaiveBayes(), Parameter(1), self.nursery))
        self.assertAlmostEqual(0.0, combined5x2t.compare(experimentPerformance1, experimentPerformance2).getPValue(), 4)
        experimentPerformance1 = mxKFoldRun.execute(Experiment(NaiveBayes(), Parameter(1), self.car))
        experimentPerformance2 = mxKFoldRun.execute(Experiment(Bagging(), BaggingParameter(1, 50), self.car))
        self.assertAlmostEqual(0.00043, combined5x2t.compare(experimentPerformance1, experimentPerformance2).getPValue(), 5)
        experimentPerformance1 = mxKFoldRun.execute(Experiment(Knn(), KnnParameter(1, 3, EuclidianDistance()), self.bupa))
        experimentPerformance2 = mxKFoldRun.execute(Experiment(Lda(), Parameter(1), self.bupa))
        self.assertAlmostEqual(0.0663, combined5x2t.compare(experimentPerformance1, experimentPerformance2).getPValue(), 4)


if __name__ == '__main__':
    unittest.main()

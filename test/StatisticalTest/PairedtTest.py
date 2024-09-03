import unittest

from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRun import KFoldRun
from Classification.Model.BaggingModel import BaggingModel
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Model.DummyModel import DummyModel
from Classification.Model.KnnModel import KnnModel
from Classification.Model.LdaModel import LdaModel
from Classification.Model.LinearPerceptronModel import LinearPerceptronModel
from Classification.Model.NaiveBayesModel import NaiveBayesModel
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
        experimentPerformance1 = kFoldRun.execute(Experiment(DecisionTree(), C45Parameter(1, True, 0.2), self.iris))
        experimentPerformance2 = kFoldRun.execute(Experiment(LinearPerceptronModel(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.iris))
        self.assertAlmostEqual(0.278, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 3)
        experimentPerformance1 = kFoldRun.execute(Experiment(DecisionTree(), C45Parameter(1, True, 0.2), self.tictactoe))
        experimentPerformance2 = kFoldRun.execute(Experiment(BaggingModel(), BaggingParameter(1, 50), self.tictactoe))
        self.assertAlmostEqual(0.00000692, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 7)
        experimentPerformance1 = kFoldRun.execute(Experiment(LdaModel(), Parameter(1), self.dermatology))
        experimentPerformance2 = kFoldRun.execute(Experiment(LinearPerceptronModel(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.dermatology))
        self.assertAlmostEqual(0.7842, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 4)
        experimentPerformance1 = kFoldRun.execute(Experiment(DummyModel(), Parameter(1), self.nursery))
        experimentPerformance2 = kFoldRun.execute(Experiment(NaiveBayesModel(), Parameter(1), self.nursery))
        self.assertAlmostEqual(0.0, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 4)
        experimentPerformance1 = kFoldRun.execute(Experiment(NaiveBayesModel(), Parameter(1), self.car))
        experimentPerformance2 = kFoldRun.execute(Experiment(BaggingModel(), BaggingParameter(1, 50), self.car))
        self.assertAlmostEqual(0.00000336, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 7)
        experimentPerformance1 = kFoldRun.execute(Experiment(KnnModel(), KnnParameter(1, 3, EuclidianDistance()), self.bupa))
        experimentPerformance2 = kFoldRun.execute(Experiment(LdaModel(), Parameter(1), self.bupa))
        self.assertAlmostEqual(0.1640, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 4)


if __name__ == '__main__':
    unittest.main()

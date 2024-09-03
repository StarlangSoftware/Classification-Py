import unittest

from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.MxKFoldRun import MxKFoldRun
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
from test.Classifier.ClassifierTest import ClassifierTest


class MxKFoldRunTest(ClassifierTest):

    def test_Execute(self):
        mxKFoldRun = MxKFoldRun(5, 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(DecisionTree(), C45Parameter(1, True, 0.2), self.iris))
        self.assertAlmostEqual(5.47, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(DecisionTree(), C45Parameter(1, True, 0.2), self.tictactoe))
        self.assertAlmostEqual(23.51, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(KnnModel(), KnnParameter(1, 3, EuclidianDistance()), self.bupa))
        self.assertAlmostEqual(37.05, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(KnnModel(), KnnParameter(1, 3, EuclidianDistance()), self.dermatology))
        self.assertAlmostEqual(15.41, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(LdaModel(), Parameter(1), self.bupa))
        self.assertAlmostEqual(34.72, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(LdaModel(), Parameter(1), self.dermatology))
        self.assertAlmostEqual(4.04, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(LinearPerceptronModel(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.iris))
        self.assertAlmostEqual(5.2, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(LinearPerceptronModel(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.dermatology))
        self.assertAlmostEqual(5.46, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(NaiveBayesModel(), Parameter(1), self.car))
        self.assertAlmostEqual(16.52, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(NaiveBayesModel(), Parameter(1), self.nursery))
        self.assertAlmostEqual(9.80, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(BaggingModel(), BaggingParameter(1, 50), self.tictactoe))
        self.assertAlmostEqual(8.77, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(BaggingModel(), BaggingParameter(1, 50), self.car))
        self.assertAlmostEqual(9.77, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(DummyModel(), Parameter(1), self.nursery))
        self.assertAlmostEqual(67.09, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = mxKFoldRun.execute(Experiment(DummyModel(), Parameter(1), self.iris))
        self.assertAlmostEqual(70.53, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()

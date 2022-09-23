from abc import abstractmethod
from Classification.Performance.ExperimentPerformance import ExperimentPerformance
from Classification.StatisticalTest.StatisticalTestResult import StatisticalTestResult
from Classification.StatisticalTest.StatisticalTestResultType import StatisticalTestResultType


class PairedTest(object):

    @abstractmethod
    def compare(self,
                classifier1: ExperimentPerformance,
                classifier2: ExperimentPerformance) -> StatisticalTestResult:
        pass

    def compareWithAlpha(self,
                         classifier1: ExperimentPerformance,
                         classifier2: ExperimentPerformance,
                         alpha: float) -> int:
        test_result1 = self.compare(classifier1, classifier2)
        test_result2 = self.compare(classifier2, classifier1)
        test_result_type1 = test_result1.oneTailed(alpha)
        test_result_type2 = test_result2.oneTailed(alpha)
        if test_result_type1 is StatisticalTestResultType.REJECT:
            return 1
        else:
            if test_result_type2 is StatisticalTestResultType.REJECT:
                return -1
            else:
                return 0

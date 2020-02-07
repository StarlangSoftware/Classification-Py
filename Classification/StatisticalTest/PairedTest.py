from abc import abstractmethod
from Classification.Performance.ExperimentPerformance import ExperimentPerformance
from Classification.StatisticalTest.StatisticalTestResult import StatisticalTestResult
from Classification.StatisticalTest.StatisticalTestResultType import StatisticalTestResultType


class PairedTest(object):

    @abstractmethod
    def compare(self, classifier1: ExperimentPerformance, classifier2: ExperimentPerformance) -> StatisticalTestResult:
        pass

    def compareWithAlpha(self, classifier1: ExperimentPerformance, classifier2: ExperimentPerformance, alpha: float) \
            -> int:
        testResult1 = self.compare(classifier1, classifier2)
        testResult2 = self.compare(classifier2, classifier1)
        testResultType1 = testResult1.oneTailed(alpha)
        testResultType2 = testResult2.oneTailed(alpha)
        if testResultType1 is StatisticalTestResultType.REJECT:
            return 1
        else:
            if testResultType2 is StatisticalTestResultType.REJECT:
                return -1
            else:
                return 0

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
        """
        Compares two classification algorithms based on their performances (accuracy or error rate). The method first
        checks the null hypothesis mu1 < mu2, if the test rejects this null hypothesis with alpha level of confidence, it
        decides mu1 > mu2. The algorithm then checks the null hypothesis mu1 > mu2, if the test rejects that null
        hypothesis with alpha level of confidence, if decides mu1 < mu2. If none of the two tests are rejected, it can not
        make a decision about the performances of algorithms.
        :param classifier1: Performance (error rate or accuracy) results of the first classifier.
        :param classifier2: Performance (error rate or accuracy) results of the second classifier.
        :param alpha: Alpha level defined for the statistical test.
        :return: 1 if the performance of the first algorithm is larger than the second algorithm, -1 if the performance of
        the second algorithm is larger than the first algorithm, 0 if they have similar performance.
        """
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

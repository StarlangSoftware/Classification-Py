from Classification.StatisticalTest.PairedTest import PairedTest
from Classification.Performance.ExperimentPerformance import ExperimentPerformance
from Classification.StatisticalTest.StatisticalTestNotApplicable import StatisticalTestNotApplicable
from Classification.StatisticalTest.StatisticalTestResult import StatisticalTestResult
import math


class Sign(PairedTest):

    def __binomial(self, m: int, n: int) -> int:
        if n == 0 or m == n:
            return 1
        else:
            return math.factorial(m) // (math.factorial(n) * math.factorial(m - n))

    def compare(self, classifier1: ExperimentPerformance, classifier2: ExperimentPerformance) -> StatisticalTestResult:
        if classifier1.numberOfExperiments() != classifier2.numberOfExperiments():
            raise StatisticalTestNotApplicable("In order to apply a paired test, you need to have the same number of "
                                               "experiments in both algorithms.")
        plus = 0
        minus = 0
        for i in range(classifier1.numberOfExperiments()):
            if classifier1.getErrorRate(i) < classifier2.getErrorRate(i):
                plus = plus + 1
            else:
                if classifier1.getErrorRate(i) > classifier2.getErrorRate(i):
                    minus = minus + 1
        total = plus + minus
        pValue = 0
        if total == 0:
            raise StatisticalTestNotApplicable("Variance is 0.")
        for i in range(plus + 1):
            pValue += self.__binomial(total, i) / math.pow(2, total)
        return StatisticalTestResult(pValue, False)

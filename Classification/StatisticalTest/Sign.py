from Classification.StatisticalTest.PairedTest import PairedTest
from Classification.Performance.ExperimentPerformance import ExperimentPerformance
from Classification.StatisticalTest.StatisticalTestNotApplicable import StatisticalTestNotApplicable
from Classification.StatisticalTest.StatisticalTestResult import StatisticalTestResult
import math


class Sign(PairedTest):

    def __binomial(self,
                   m: int,
                   n: int) -> int:
        """
        Calculates m of n that is C(n, m)
        :param m: m in C(m, n)
        :param n: n in C(m, n)
        :return: C(m, n)
        """
        if n == 0 or m == n:
            return 1
        else:
            return math.factorial(m) // (math.factorial(n) * math.factorial(m - n))

    def compare(self,
                classifier1: ExperimentPerformance,
                classifier2: ExperimentPerformance) -> StatisticalTestResult:
        """
        Compares two classification algorithms based on their performances (accuracy or error rate) using sign test.
        :param classifier1: Performance (error rate or accuracy) results of the first classifier.
        :param classifier2: Performance (error rate or accuracy) results of the second classifier.
        :return: Statistical test result of the comparison.
        """
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
        p_value = 0
        if total == 0:
            raise StatisticalTestNotApplicable("Variance is 0.")
        for i in range(plus + 1):
            p_value += self.__binomial(total, i) / math.pow(2, total)
        return StatisticalTestResult(p_value, False)

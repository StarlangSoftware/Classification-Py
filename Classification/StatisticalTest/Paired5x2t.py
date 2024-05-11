from Classification.StatisticalTest.PairedTest import PairedTest
from Classification.Performance.ExperimentPerformance import ExperimentPerformance
from Classification.StatisticalTest.StatisticalTestNotApplicable import StatisticalTestNotApplicable
from Classification.StatisticalTest.StatisticalTestResult import StatisticalTestResult
from Math.Distribution import Distribution
import math


class Paired5x2t(PairedTest):

    def __testStatistic(self,
                        classifier1: ExperimentPerformance,
                        classifier2: ExperimentPerformance):
        """
        Calculates the test statistic of the 5x2 t test.
        :param classifier1: Performance (error rate or accuracy) results of the first classifier.
        :param classifier2: Performance (error rate or accuracy) results of the second classifier.
        :return: Given the performances of two classifiers, the test statistic of the 5x2 t test.
        """
        if classifier1.numberOfExperiments() != classifier2.numberOfExperiments():
            raise StatisticalTestNotApplicable("In order to apply a paired test, you need to have the same number of "
                                               "experiments in both algorithms.")
        if classifier1.numberOfExperiments() != 10:
            raise StatisticalTestNotApplicable("In order to apply a 5x2 test, you need to have 10 experiments.")
        difference = []
        for i in range(classifier1.numberOfExperiments()):
            difference.append(classifier1.getErrorRate(i) - classifier2.getErrorRate(i))
        denominator = 0
        for i in range(classifier1.numberOfExperiments() // 2):
            mean = (difference[2 * i] + difference[2 * i + 1]) / 2
            variance = (difference[2 * i] - mean) * (difference[2 * i] - mean) + (difference[2 * i + 1] - mean) \
                       * (difference[2 * i + 1] - mean)
            denominator += variance
        denominator = math.sqrt(denominator / 5)
        if denominator == 0:
            raise StatisticalTestNotApplicable("Variance is 0.")
        return difference[0] / denominator

    def compare(self,
                classifier1: ExperimentPerformance,
                classifier2: ExperimentPerformance) -> StatisticalTestResult:
        """
        Compares two classification algorithms based on their performances (accuracy or error rate) using 5x2 t test.
        :param classifier1: Performance (error rate or accuracy) results of the first classifier.
        :param classifier2: Performance (error rate or accuracy) results of the second classifier.
        :return: Statistical test result of the comparison.
        """
        statistic = self.__testStatistic(classifier1, classifier2)
        degree_of_freedom = classifier1.numberOfExperiments() // 2
        return StatisticalTestResult(Distribution.tDistribution(statistic, degree_of_freedom), False)

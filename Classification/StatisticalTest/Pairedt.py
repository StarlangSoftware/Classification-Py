from Classification.StatisticalTest.PairedTest import PairedTest
from Classification.Performance.ExperimentPerformance import ExperimentPerformance
from Classification.StatisticalTest.StatisticalTestNotApplicable import StatisticalTestNotApplicable
from Classification.StatisticalTest.StatisticalTestResult import StatisticalTestResult
from Math.Distribution import Distribution
import math


class Pairedt(PairedTest):

    def __testStatistic(self,
                        classifier1: ExperimentPerformance,
                        classifier2: ExperimentPerformance):
        """
        Calculates the test statistic of the paired t test.
        :param classifier1: Performance (error rate or accuracy) results of the first classifier.
        :param classifier2: Performance (error rate or accuracy) results of the second classifier.
        :return: Given the performances of two classifiers, the test statistic of the paired t test.
        """
        if classifier1.numberOfExperiments() != classifier2.numberOfExperiments():
            raise StatisticalTestNotApplicable("In order to apply a paired test, you need to have the same number of "
                                               "experiments in both algorithms.")
        difference = []
        total = 0
        for i in range(classifier1.numberOfExperiments()):
            difference.append(classifier1.getErrorRate(i) - classifier2.getErrorRate(i))
            total += difference[i]
        mean = total / classifier1.numberOfExperiments()
        total = 0
        for i in range(classifier1.numberOfExperiments()):
            total += (difference[i] - mean) * (difference[i] - mean)
        standard_deviation = math.sqrt(total / (classifier1.numberOfExperiments() - 1))
        if standard_deviation == 0:
            raise StatisticalTestNotApplicable("Variance is 0.")
        return math.sqrt(classifier1.numberOfExperiments()) * mean / standard_deviation

    def compare(self,
                classifier1: ExperimentPerformance,
                classifier2: ExperimentPerformance) -> StatisticalTestResult:
        """
        Compares two classification algorithms based on their performances (accuracy or error rate) using paired t test.
        :param classifier1: Performance (error rate or accuracy) results of the first classifier.
        :param classifier2: Performance (error rate or accuracy) results of the second classifier.
        :return: Statistical test result of the comparison.
        """
        statistic = self.__testStatistic(classifier1, classifier2)
        degree_of_freedom = classifier1.numberOfExperiments() - 1
        return StatisticalTestResult(Distribution.tDistribution(statistic, degree_of_freedom), False)

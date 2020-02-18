from Classification.StatisticalTest.PairedTest import PairedTest
from Classification.Performance.ExperimentPerformance import ExperimentPerformance
from Classification.StatisticalTest.StatisticalTestNotApplicable import StatisticalTestNotApplicable
from Classification.StatisticalTest.StatisticalTestResult import StatisticalTestResult
from Math.Distribution import Distribution
import math


class Pairedt(PairedTest):

    def __testStatistic(self, classifier1: ExperimentPerformance, classifier2: ExperimentPerformance):
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
        standardDeviation = math.sqrt(total / (classifier1.numberOfExperiments() - 1))
        if standardDeviation == 0:
            raise StatisticalTestNotApplicable("Variance is 0.")
        return math.sqrt(classifier1.numberOfExperiments()) * mean / standardDeviation

    def compare(self, classifier1: ExperimentPerformance, classifier2: ExperimentPerformance) -> StatisticalTestResult:
        statistic = self.__testStatistic(classifier1, classifier2)
        degreeOfFreedom = classifier1.numberOfExperiments() - 1
        return StatisticalTestResult(Distribution.tDistribution(statistic, degreeOfFreedom), False)

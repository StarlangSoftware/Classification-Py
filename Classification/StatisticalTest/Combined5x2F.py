from Classification.StatisticalTest.PairedTest import PairedTest
from Classification.Performance.ExperimentPerformance import ExperimentPerformance
from Classification.StatisticalTest.StatisticalTestNotApplicable import StatisticalTestNotApplicable
from Classification.StatisticalTest.StatisticalTestResult import StatisticalTestResult
from Math.Distribution import Distribution


class Combined5x2F(PairedTest):

    def __testStatistic(self, classifier1: ExperimentPerformance, classifier2: ExperimentPerformance):
        if classifier1.numberOfExperiments() != classifier2.numberOfExperiments():
            raise StatisticalTestNotApplicable("In order to apply a paired test, you need to have the same number of "
                                               "experiments in both algorithms.")
        if classifier1.numberOfExperiments() != 10:
            raise StatisticalTestNotApplicable("In order to apply a 5x2 test, you need to have 10 experiments.")
        numerator = 0
        difference = []
        for i in range(classifier1.numberOfExperiments()):
            difference.append(classifier1.getErrorRate(i) - classifier2.getErrorRate(i))
            numerator += difference[i] * difference[i]
        denominator = 0
        for i in range(classifier1.numberOfExperiments() // 2):
            mean = (difference[2 * i] + difference[2 * i + 1]) / 2
            variance = (difference[2 * i] - mean) * (difference[2 * i] - mean) + (difference[2 * i + 1] - mean) \
                       * (difference[2 * i + 1] - mean)
            denominator += variance
        denominator *= 2
        if denominator == 0:
            raise StatisticalTestNotApplicable("Variance is 0.")
        return numerator / denominator

    def compare(self, classifier1: ExperimentPerformance, classifier2: ExperimentPerformance) -> StatisticalTestResult:
        statistic = self.__testStatistic(classifier1, classifier2)
        degreeOfFreedom1 = classifier1.numberOfExperiments()
        degreeOfFreedom2 = classifier2.numberOfExperiments() // 2
        return StatisticalTestResult(Distribution.fDistribution(statistic, degreeOfFreedom1, degreeOfFreedom2), True)

from Classification.StatisticalTest.StatisticalTestNotApplicable import StatisticalTestNotApplicable
from Classification.StatisticalTest.StatisticalTestResultType import StatisticalTestResultType


class StatisticalTestResult(object):

    def __init__(self, pValue: float, onlyTwoTailed: bool):
        self.pValue = pValue
        self.onlyTwoTailed = onlyTwoTailed

    def oneTailed(self, alpha: float) -> StatisticalTestResultType:
        if self.onlyTwoTailed:
            raise StatisticalTestNotApplicable("One tailed option is not available for this test. The distribution is one tailed distribution.")
        if self.pValue < alpha:
            return StatisticalTestResultType.REJECT
        else:
            return StatisticalTestResultType.FAILED_TO_REJECT

    def twoTailed(self, alpha: float) -> StatisticalTestResultType:
        if self.onlyTwoTailed:
            if self.pValue < alpha:
                return StatisticalTestResultType.REJECT
            else:
                return StatisticalTestResultType.FAILED_TO_REJECT
        else:
            if self.pValue < alpha / 2 or self.pValue > 1 - alpha / 2:
                return StatisticalTestResultType.REJECT

    def getPValue(self) -> float:
        return self.pValue
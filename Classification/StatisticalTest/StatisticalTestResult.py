from Classification.StatisticalTest.StatisticalTestNotApplicable import StatisticalTestNotApplicable
from Classification.StatisticalTest.StatisticalTestResultType import StatisticalTestResultType


class StatisticalTestResult(object):

    __pValue: float
    __onlyTwoTailed: bool

    def __init__(self, pValue: float, onlyTwoTailed: bool):
        self.__pValue = pValue
        self.__onlyTwoTailed = onlyTwoTailed

    def oneTailed(self, alpha: float) -> StatisticalTestResultType:
        if self.__onlyTwoTailed:
            raise StatisticalTestNotApplicable("One tailed option is not available for this test. The distribution is "
                                               "one tailed distribution.")
        if self.__pValue < alpha:
            return StatisticalTestResultType.REJECT
        else:
            return StatisticalTestResultType.FAILED_TO_REJECT

    def twoTailed(self, alpha: float) -> StatisticalTestResultType:
        if self.__onlyTwoTailed:
            if self.__pValue < alpha:
                return StatisticalTestResultType.REJECT
            else:
                return StatisticalTestResultType.FAILED_TO_REJECT
        else:
            if self.__pValue < alpha / 2 or self.__pValue > 1 - alpha / 2:
                return StatisticalTestResultType.REJECT

    def getPValue(self) -> float:
        return self.__pValue

from Classification.StatisticalTest.StatisticalTestNotApplicable import StatisticalTestNotApplicable
from Classification.StatisticalTest.StatisticalTestResultType import StatisticalTestResultType


class StatisticalTestResult(object):

    __pValue: float
    __only_two_tailed: bool

    def __init__(self,
                 pValue: float,
                 onlyTwoTailed: bool):
        """
        Constructor of the StatisticalTestResult. It sets the attribute values.
        :param pValue: p value of the statistical test result
        :param onlyTwoTailed: True, if this test applicable only two tailed tests, false otherwise.
        """
        self.__pValue = pValue
        self.__only_two_tailed = onlyTwoTailed

    def oneTailed(self, alpha: float) -> StatisticalTestResultType:
        """
        Returns reject or failed to reject, depending on the alpha level and p value of the statistical test that checks
        one tailed null hypothesis such as mu1 < mu2. If p value is less than the alpha level, the test rejects the null
        hypothesis. Otherwise, it fails to reject the null hypothesis.
        :param alpha: Alpha level of the test
        :return: If p value is less than the alpha level, the test rejects the null hypothesis. Otherwise, it fails to
        reject the null hypothesis.
        """
        if self.__only_two_tailed:
            raise StatisticalTestNotApplicable("One tailed option is not available for this test. The distribution is "
                                               "one tailed distribution.")
        if self.__pValue < alpha:
            return StatisticalTestResultType.REJECT
        else:
            return StatisticalTestResultType.FAILED_TO_REJECT

    def twoTailed(self, alpha: float) -> StatisticalTestResultType:
        """
        Returns reject or failed to reject, depending on the alpha level and p value of the statistical test that checks
        one tailed null hypothesis such as mu1 < mu2 or two tailed null hypothesis such as mu1 = mu2. If the null
        hypothesis is two tailed, and p value is less than the alpha level, the test rejects the null hypothesis.
        Otherwise, it fails to reject the null hypothesis. If the null  hypothesis is one tailed, and p value is less
        than alpha / 2 or p value is larger than 1 - alpha / 2, the test  rejects the null  hypothesis. Otherwise, it
        fails to reject the null hypothesis.
        :param alpha: Alpha level of the test
        :return: If the null  hypothesis is two tailed, and p value is less than the alpha level, the test rejects the
        null hypothesis.  Otherwise, it fails to reject the null hypothesis. If the null  hypothesis is one tailed, and
        p value is less  than alpha / 2 or p value is larger than 1 - alpha / 2, the test  rejects the null  hypothesis.
        Otherwise, it  fails to reject the null hypothesis.
        """
        if self.__only_two_tailed:
            if self.__pValue < alpha:
                return StatisticalTestResultType.REJECT
            else:
                return StatisticalTestResultType.FAILED_TO_REJECT
        else:
            if self.__pValue < alpha / 2 or self.__pValue > 1 - alpha / 2:
                return StatisticalTestResultType.REJECT

    def getPValue(self) -> float:
        """
        Returns the p value of the statistical test result
        :return: p value of the statistical test result
        """
        return self.__pValue

from enum import Enum, auto


"""
 * Enumerator class for statistical test results.
"""
class StatisticalTestResultType(Enum):
    FAILED_TO_REJECT = auto()
    REJECT = auto()

from __future__ import annotations
from Classification.Performance.Performance import Performance
from Classification.Performance.ClassificationPerformance import ClassificationPerformance
from Classification.Performance.DetailedClassificationPerformance import DetailedClassificationPerformance
import math


class ExperimentPerformance:

    """
    A constructor which creates a new list of Performance as results.
    """
    def __init__(self):
        self.results = []
        self.containsDetails = True
        self.classification = True

    def __gt__(self, other) -> bool:
        accuracy1 = self.meanClassificationPerformance().getAccuracy()
        accuracy2 = other.meanClassificationPerformance().getAccuracy()
        return accuracy1 > accuracy2

    def __lt__(self, other) -> bool:
        accuracy1 = self.meanClassificationPerformance().getAccuracy()
        accuracy2 = other.meanClassificationPerformance().getAccuracy()
        return accuracy1 < accuracy2

    """
    A constructor that takes a file name as an input and takes the inputs from that file assigns these inputs to the 
    errorRate and adds them to the results list as a new Performance.
    
    PARAMETERS
    ----------     
    fileName : str
        String input.
    """
    def initWithFile(self, fileName: str):
        self.containsDetails = False
        input = open(fileName, "r")
        lines = input.readlines()
        for line in lines:
            self.results.append(Performance(float(line)))

    """
    The add method takes a Performance as an input and adds it to the results list.

    PARAMETERS
    ----------
    performance : Performance
        Performance input.
    """
    def add(self, performance: Performance):
        if not isinstance(performance, DetailedClassificationPerformance):
            self.containsDetails = False
        if not isinstance(performance, ClassificationPerformance):
            self.classification = False
        self.results.append(performance)

    """
    The numberOfExperiments method returns the size of the results {@link ArrayList}.

    RETURNS
    -------
    int
        The results list.
    """
    def numberOfExperiments(self) -> int:
        return len(self.results)

    """
    The getErrorRate method takes an index as an input and returns the errorRate at given index of results list.

    PARAMETERS
    ----------
    index : int
        Index of results list to retrieve.
        
    RETURNS
    -------
    float
        The errorRate at given index of results list.
    """
    def getErrorRate(self, index: int) -> float:
        return self.results[index]

    """
    The getAccuracy method takes an index as an input. It returns the accuracy of a Performance at given index 
    of results list.

    PARAMETERS
    ----------
    index : int
        Index of results list to retrieve.
        
    RETURNS
    -------
    float
        The accuracy of a Performance at given index of results list.
    """
    def getAccuracy(self, index: int) -> float:
        return self.results[index].getAccuracy()

    """
    The meanPerformance method loops through the performances of results list and sums up the errorRates of each then
    returns a new Performance with the mean of that summation.

    RETURNS
    -------
    Performance
        A new Performance with the mean of the summation of errorRates.
    """
    def meanPerformance(self) -> Performance:
        sumError = 0
        for performance in self.results:
            sumError += performance.getErrorRate()
        return Performance(sumError / len(self.results))

    """
    The meanClassificationPerformance method loops through the performances of results list and sums up 
    the accuracy of each classification performance, then returns a new classificationPerformance with the mean of 
    that summation.

    RETURNS
    -------
    ClassificationPerformance
        A new classificationPerformance with the mean of that summation.
    """
    def meanClassificationPerformance(self) -> ClassificationPerformance:
        if len(self.results) == 0 or not self.classification:
            return None
        sumAccuracy = 0
        for performance in self.results:
            sumAccuracy += performance.getAccuracy()
        return ClassificationPerformance(sumAccuracy / len(self.results))

    """
    The meanDetailedPerformance method gets the first confusion matrix of results list.
    Then, it adds new confusion matrices as the DetailedClassificationPerformance of other elements of results 
    ArrayList' confusion matrices as a DetailedClassificationPerformance.

    RETURNS
    -------
    DetailedCassificationPerformance
        A new DetailedClassificationPerformance with the ConfusionMatrix sum.
    """
    def meanDetailedPerformance(self) -> DetailedClassificationPerformance:
        if len(self.results) == 0 or not self.containsDetails:
            return None
        sum = self.results[0].getConfusionMatrix()
        for i in range(1, len(self.results)):
            sum.addConfusionMatrix(self.results[i].getConfusionMatrix())
        return DetailedClassificationPerformance(sum)

    """
    The standardDeviationPerformance method loops through the Performances of results list and returns
    a new Performance with the standard deviation.

    RETURNS
    -------
    Performance
        A new Performance with the standard deviation.
    """
    def standardDeviationPerformance(self) -> Performance:
        sumErrorRate = 0
        averagePerformance = self.meanPerformance()
        for performance in self.results:
            sumErrorRate += math.pow(performance.getErrorRate() - averagePerformance.getErrorRate(), 2)
        return Performance(math.sqrt(sumErrorRate / (len(self.results) - 1)))

    """
    The standardDeviationClassificationPerformance method loops through the Performances of results list and
    returns a new ClassificationPerformance with standard deviation.

    RETURNS
    -------
    ClassificationPerformance
        A new ClassificationPerformance with standard deviation.
    """
    def standardDeviationClassificationPerformance(self) -> ClassificationPerformance:
        if len(self.results) == 0 or not self.classification:
            return None
        sumAccuracy = 0
        sumErrorRate = 0
        averageClassificationPerformance = self.meanClassificationPerformance()
        for performance in self.results:
            sumAccuracy += math.pow(performance.getAccuracy() - averageClassificationPerformance.getAccuracy(), 2)
            sumErrorRate += math.pow(performance.getErrorRate() - averageClassificationPerformance.getErrorRate(), 2)
        return ClassificationPerformance(math.sqrt(sumAccuracy / (len(self.results) - 1)))

    """
    The isBetter method  takes an ExperimentPerformance as an input and returns true if the result of compareTo method 
    is positive and false otherwise.

    PARAMETERS
    ----------
    experimentPerformance : ExperimentPerformance
        ExperimentPerformance input.
        
    RETURNS
    -------
    bool
        True if the experiment performance is better than the given experiment performance.
    """
    def isBetter(self, experimentPerformance: ExperimentPerformance) -> bool:
        return self > experimentPerformance
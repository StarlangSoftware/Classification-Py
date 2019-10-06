from __future__ import annotations
from DataStructure.CounterHashMap import CounterHashMap


class ConfusionMatrix:

    """
    Constructor that sets the class labels list and creates new dictionary matrix

    PARAMETERS
    ----------
    classLabels : list
        list of String.
    """
    def __init__(self, classLabels: list):
        self.classLabels = classLabels
        self.matrix = {}

    """
    The classify method takes two Strings; actual class and predicted class as inputs. If the matrix dictionary contains
    given actual class String as a key, it then assigns the corresponding object of that key to a CounterHashMap, if not
    it creates a new CounterHashMap. Then, it puts the given predicted class String to the counterHashMap and
    also put this counterHashMap to the matrix dictionary together with the given actual class String.

    PARAMETERS
    ----------
    actualClass : str
        String input actual class.
    predictedClass : str
        String input predicted class.
    """
    def classify(self, actualClass: str, predictedClass: str):
        if actualClass in self.matrix:
            counterHashMap = self.matrix[actualClass]
        else:
            counterHashMap = CounterHashMap()
        counterHashMap.put(predictedClass)
        self.matrix[actualClass] = counterHashMap

    """
    The addConfusionMatrix method takes a ConfusionMatrix as an input and loops through actual classes of that 
    dictionary and initially gets one row at a time. Then it puts the current row to the matrix dictionary together 
    with the actual class string.

    PARAMETERS
    ----------
    confusionMatrix : ConfusionMatrix
        ConfusionMatrix input.
    """
    def addConfusionMatrix(self, confusionMatrix: ConfusionMatrix):
        for actualClass in confusionMatrix.matrix:
            rowToBeAdded = confusionMatrix.matrix[actualClass]
            if actualClass in self.matrix:
                currentRow = self.matrix[actualClass]
                currentRow.add(rowToBeAdded)
                self.matrix[actualClass] = currentRow
            else:
                self.matrix[actualClass] = rowToBeAdded

    """
    The sumOfElements method loops through the keys in matrix dictionary and returns the summation of all the values 
    of the keys. I.e: TP+TN+FP+FN.

    RETURNS
    -------
    float
        The summation of values.
    """
    def sumOfElements(self) -> float:
        result = 0
        for actualClass in self.matrix:
            result += self.matrix[actualClass]
        return result

    """
    The trace method loops through the keys in matrix dictionary and if the current key contains the actual key,
    it accumulates the corresponding values. I.e: TP+TN.

    RETURNS
    -------
    float
        Summation of values.
    """
    def trace(self) -> float:
        result = 0
        for actualClass in self.matrix:
            if actualClass in self.matrix[actualClass]:
                result += self.matrix[actualClass][actualClass]
        return result

    """
    The columnSum method takes a String predicted class as input, and loops through the keys in matrix dictionary.
    If the current key contains the predicted class String, it accumulates the corresponding values. I.e: TP+FP.

    PARAMETERS
    ----------
    predictedClass : str
        String input predicted class.
        
    RETURNS
    -------
    float
        Summation of values.
    """
    def columnSum(self, predictedClass: str) -> float:
        result = 0
        for actualClass in self.matrix:
            if predictedClass in self.matrix[actualClass]:
                result += self.matrix[actualClass][predictedClass]
        return result

    """
    The getAccuracy method returns the result of  TP+TN / TP+TN+FP+FN

    RETURNS
    -------
    list
        the result of  TP+TN / TP+TN+FP+FN
    """
    def getAccuracy(self) -> float:
        return self.trace() / self.sumOfElements()

    """
    The precision method loops through the class labels and returns the resulting Array which has the result of TP/FP+TP.
     *
     * @return The result of TP/FP+TP.
    """
    def precision(self) -> list:
        result = []
        for i in range(len(self.classLabels)):
            actualClass = self.classLabels[i]
            if actualClass in self.matrix:
                result.append(self.matrix[actualClass][actualClass] / self.columnSum(actualClass))
        return result

    """
    The recall method loops through the class labels and returns the resulting Array which has the result of TP/FN+TP.

    RETURNS
    -------
    list
        The result of TP/FN+TP.
    """
    def recall(self) -> list:
        result = []
        for i in range(len(self.classLabels)):
            actualClass = self.classLabels[i]
            if actualClass in self.matrix:
                result.append(self.matrix[actualClass][actualClass] / self.matrix[actualClass].sumOfCounts())
        return result

    """
    The fMeasure method loops through the class labels and returns the resulting Array which has the average of
    recall and precision.

    RETURNS
    -------
    list
        The average of recall and precision.
    """
    def fMeasure(self) -> list:
        precision = self.precision()
        recall = self.recall()
        result = []
        for i in range(len(self.classLabels)):
            result.append(2 / (1 / precision[i] + 1 / recall[i]))
        return result

    """
    The weightedFMeasure method loops through the class labels and returns the resulting Array which has the weighted 
    average of recall and precision.

    RETURNS
    -------
    float
        The weighted average of recall and precision.
    """
    def weightedFMeasure(self) -> float:
        fMeasure = self.fMeasure()
        sum = 0
        for i in range(len(self.classLabels)):
            actualClass = self.classLabels[i]
            sum += fMeasure[i] * self.matrix[actualClass].sumOfCounts()
        return sum / self.sumOfElements()
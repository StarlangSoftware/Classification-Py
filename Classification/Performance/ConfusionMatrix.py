from __future__ import annotations
from DataStructure.CounterHashMap import CounterHashMap


class ConfusionMatrix:

    __matrix: dict
    __classLabels: list

    def __init__(self, classLabels: list):
        """
        Constructor that sets the class labels list and creates new dictionary matrix

        PARAMETERS
        ----------
        classLabels : list
            list of String.
        """
        self.__classLabels = classLabels
        self.__matrix = {}

    def classify(self, actualClass: str, predictedClass: str):
        """
        The classify method takes two Strings; actual class and predicted class as inputs. If the matrix dictionary
        contains given actual class String as a key, it then assigns the corresponding object of that key to a
        CounterHashMap, if not it creates a new CounterHashMap. Then, it puts the given predicted class String to the
        counterHashMap and also put this counterHashMap to the matrix dictionary together with the given actual class
        String.

        PARAMETERS
        ----------
        actualClass : str
            String input actual class.
        predictedClass : str
            String input predicted class.
        """
        if actualClass in self.__matrix:
            counterHashMap = self.__matrix[actualClass]
        else:
            counterHashMap = CounterHashMap()
        counterHashMap.put(predictedClass)
        self.__matrix[actualClass] = counterHashMap

    def addConfusionMatrix(self, confusionMatrix: ConfusionMatrix):
        """
        The addConfusionMatrix method takes a ConfusionMatrix as an input and loops through actual classes of that
        dictionary and initially gets one row at a time. Then it puts the current row to the matrix dictionary together
        with the actual class string.

        PARAMETERS
        ----------
        confusionMatrix : ConfusionMatrix
            ConfusionMatrix input.
        """
        for actualClass in confusionMatrix.__matrix:
            rowToBeAdded = confusionMatrix.__matrix[actualClass]
            if actualClass in self.__matrix:
                currentRow = self.__matrix[actualClass]
                currentRow.add(rowToBeAdded)
                self.__matrix[actualClass] = currentRow
            else:
                self.__matrix[actualClass] = rowToBeAdded

    def sumOfElements(self) -> float:
        """
        The sumOfElements method loops through the keys in matrix dictionary and returns the summation of all the values
        of the keys. I.e: TP+TN+FP+FN.

        RETURNS
        -------
        float
            The summation of values.
        """
        result = 0
        for actualClass in self.__matrix:
            result += self.__matrix[actualClass].sumOfCounts()
        return result

    def trace(self) -> float:
        """
        The trace method loops through the keys in matrix dictionary and if the current key contains the actual key,
        it accumulates the corresponding values. I.e: TP+TN.

        RETURNS
        -------
        float
            Summation of values.
        """
        result = 0
        for actualClass in self.__matrix:
            if actualClass in self.__matrix[actualClass]:
                result += self.__matrix[actualClass][actualClass]
        return result

    def columnSum(self, predictedClass: str) -> float:
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
        result = 0
        for actualClass in self.__matrix:
            if predictedClass in self.__matrix[actualClass]:
                result += self.__matrix[actualClass][predictedClass]
        return result

    def getAccuracy(self) -> float:
        """
        The getAccuracy method returns the result of  TP+TN / TP+TN+FP+FN

        RETURNS
        -------
        list
            the result of  TP+TN / TP+TN+FP+FN
        """
        return self.trace() / self.sumOfElements()

    def precision(self) -> list:
        """
        The precision method loops through the class labels and returns the resulting Array which has the result of
        TP/FP+TP.

        RETURNS
        -------
        list
            The result of TP/FP+TP.
        """
        result = []
        for i in range(len(self.__classLabels)):
            actualClass = self.__classLabels[i]
            if actualClass in self.__matrix:
                result.append(self.__matrix[actualClass][actualClass] / self.columnSum(actualClass))
        return result

    def recall(self) -> list:
        """
        The recall method loops through the class labels and returns the resulting Array which has the result of
        TP/FN+TP.

        RETURNS
        -------
        list
            The result of TP/FN+TP.
        """
        result = []
        for i in range(len(self.__classLabels)):
            actualClass = self.__classLabels[i]
            if actualClass in self.__matrix:
                result.append(self.__matrix[actualClass][actualClass] / self.__matrix[actualClass].sumOfCounts())
        return result

    def fMeasure(self) -> list:
        """
        The fMeasure method loops through the class labels and returns the resulting Array which has the average of
        recall and precision.

        RETURNS
        -------
        list
            The average of recall and precision.
        """
        precision = self.precision()
        recall = self.recall()
        result = []
        for i in range(len(self.__classLabels)):
            result.append(2 / (1 / precision[i] + 1 / recall[i]))
        return result

    def weightedFMeasure(self) -> float:
        """
        The weightedFMeasure method loops through the class labels and returns the resulting Array which has the
        weighted average of recall and precision.

        RETURNS
        -------
        float
            The weighted average of recall and precision.
        """
        fMeasure = self.fMeasure()
        total = 0
        for i in range(len(self.__classLabels)):
            actualClass = self.__classLabels[i]
            total += fMeasure[i] * self.__matrix[actualClass].sumOfCounts()
        return total / self.sumOfElements()

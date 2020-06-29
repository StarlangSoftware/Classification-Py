from abc import abstractmethod

from DataStructure.CounterHashMap import CounterHashMap

from Classification.Instance.Instance import Instance


class Model(object):

    @abstractmethod
    def predict(self, instance: Instance) -> str:
        """
         An abstract predict method that takes an Instance as an input.

        PARAMETERS
        ----------
        instance : Instance
            Instance to make prediction.

        RETURNS
        -------
        str
            The class label as a String.
        """
        pass

    @staticmethod
    def getMaximum(classLabels: list) -> str:
        """
        Given an array of class labels, returns the maximum occurred one.

        PARAMETERS
        ----------
        classLabels : list
            An array of class labels.

        RETURNS
        -------
        str
            The class label that occurs most in the array of class labels (mod of class label list).
        """
        frequencies = CounterHashMap()
        for label in classLabels:
            frequencies.put(label)
        return frequencies.max()

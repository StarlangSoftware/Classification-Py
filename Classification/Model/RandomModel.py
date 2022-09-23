from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.Model.Model import Model
import random


class RandomModel(Model):

    __class_labels: list

    def __init__(self, classLabels: list, seed: int):
        """
        A constructor that sets the class labels.

        PARAMETERS
        ----------
        classLabels : list
            A List of class labels.
        seed: int
            Seed of the random function
        """
        self.__class_labels = classLabels
        random.seed(seed)

    def predict(self, instance: Instance) -> str:
        """
        The predict method gets an Instance as an input and retrieves the possible class labels as an ArrayList. Then
        selects a random number as an index and returns the class label at this selected index.

        PARAMETERS
        ----------
        instance : Instance
            Instance to make prediction.

        RETURNS
        -------
        str
            The class label at the randomly selected index.
        """
        if isinstance(instance, CompositeInstance):
            possible_class_labels = instance.getPossibleClassLabels()
            size = len(possible_class_labels)
            index = random.randint(0, size)
            return possible_class_labels[index]
        else:
            size = len(self.__class_labels)
            index = random.randrange(size)
            return self.__class_labels[index]

    def predictProbability(self, instance: Instance) -> dict:
        result = {}
        for classLabel in self.__class_labels:
            result[classLabel] = 1.0 / len(self.__class_labels)
        return result

from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.Model.Model import Model
import random


class RandomModel(Model):
    __class_labels: list
    __seed: int

    def constructor1(self,
                     classLabels: list,
                     seed: int):
        """
        A constructor that sets the class labels.

        PARAMETERS
        ----------
        classLabels : list
            A List of class labels.
        seed: int
            Seed of the random function
        """
        self.__seed = seed
        self.__class_labels = classLabels
        random.seed(seed)

    def constructor2(self, fileName: str):
        """
        Loads a random classifier model from an input model file.
        :param fileName: Model file name.
        """
        inputFile = open(fileName, mode='r', encoding='utf-8')
        self.__seed = int(inputFile.readline().strip())
        random.seed(self.__seed)
        size = int(inputFile.readline().strip())
        self.__class_labels = list()
        for i in range(size):
            self.__class_labels.append(inputFile.readline().strip())
        inputFile.close()

    def __init__(self,
                 classLabels: object,
                 seed: int = None):
        if isinstance(classLabels, list):
            self.constructor1(classLabels, seed)
        elif isinstance(classLabels, str):
            self.constructor2(classLabels)

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
        """
        Calculates the posterior probability distribution for the given instance according to random model.
        :param instance: Instance for which posterior probability distribution is calculated.
        :return: Posterior probability distribution for the given instance.
        """
        result = {}
        for classLabel in self.__class_labels:
            result[classLabel] = 1.0 / len(self.__class_labels)
        return result

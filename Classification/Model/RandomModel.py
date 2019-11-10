from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.Model.Model import Model
import random

class RandomModel(Model):

    __classLabels: list

    """
    A constructor that sets the class labels.

    PARAMETERS
    ----------
    classLabels : list
        A List of class labels.
    """
    def __init__(self, classLabels: list):
        self.__classLabels = classLabels

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
    def predict(self, instance: Instance) -> str:
        if isinstance(instance, CompositeInstance):
            possibleClassLabels = instance.getPossibleClassLabels()
            size = len(possibleClassLabels)
            index = random.randint(0, size)
            return possibleClassLabels[index]
        else:
            size = len(self.__classLabels)
            index = random.randint(0, size)
            return self.__classLabels[index]

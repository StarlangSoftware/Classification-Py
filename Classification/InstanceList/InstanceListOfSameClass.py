from Classification.InstanceList.InstanceList import InstanceList


class InstanceListOfSameClass(InstanceList):

    """
    Constructor for creating a new instance list with the same class labels.

    PARAMETERS
    ----------
    classLabel : str
        Class labels of instance list.
    """
    def __init__(self, classLabel: str):
        super().__init__()
        self.classLabel = classLabel

    """
    Accessor for the class label.

    RETURNS
    -------
    str
        Class label.
    """
    def getClassLabel(self) -> str:
        return self.classLabel
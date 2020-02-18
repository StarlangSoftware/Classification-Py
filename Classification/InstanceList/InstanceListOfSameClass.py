from Classification.InstanceList.InstanceList import InstanceList


class InstanceListOfSameClass(InstanceList):

    __classLabel: str

    def __init__(self, classLabel: str):
        """
        Constructor for creating a new instance list with the same class labels.

        PARAMETERS
        ----------
        classLabel : str
            Class labels of instance list.
        """
        super().__init__()
        self.__classLabel = classLabel

    def getClassLabel(self) -> str:
        """
        Accessor for the class label.

        RETURNS
        -------
        str
            Class label.
        """
        return self.__classLabel

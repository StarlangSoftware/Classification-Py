from Classification.Instance.Instance import Instance


class CompositeInstance(Instance):

    __possible_class_labels: list

    def __init__(self,
                 classLabel: str,
                 attributes=None,
                 possibleLabels=None):
        """
        Constructor of CompositeInstance class which takes a class label, attributes and a list of
        possible labels as inputs. It generates a new composite instance with given labels, attributes and possible
        labels.

        PARAMETERS
        ----------
        classLabel : str
            Class label of the composite instance.
        attributes : list
            Attributes of the composite instance.
        possibleLabels : list
            Possible labels of the composite instance.
        """
        super().__init__(classLabel, attributes)
        if possibleLabels is None:
            possibleLabels = []
        self.__possible_class_labels = possibleLabels

    def getPossibleClassLabels(self) -> list:
        """
        Accessor for the possible class labels.

        RETURNS
        -------
        list
            Possible class labels of the composite instance.
        """
        return self.__possible_class_labels

    def setPossibleClassLabels(self, possibleClassLabels: list):
        """
        Mutator method for possible class labels.

        PARAMETERS
        ----------
        possibleClassLabels
            Ner value of possible class labels.
        """
        self.__possible_class_labels = possibleClassLabels

    def __str__(self) -> str:
        """
        Converts possible class labels to {@link String}.

        RETURNS
        -------
        str
            String representation of possible class labels.
        """
        result = super().__str__()
        for possible_class_label in self.__possible_class_labels:
            result = result + ";" + possible_class_label
        return result

from Classification.Instance.Instance import Instance


class CompositeInstance(Instance):

    """
    Constructor of CompositeInstance class which takes a class label, attributes and a list of
    possible labels as inputs. It generates a new composite instance with given labels, attributes and possible labels.

    PARAMETERS
    ----------
    classLabel : str
        Class label of the composite instance.
    attributes : list
        Attributes of the composite instance.
    possibleClassLabels : list
        Possible labels of the composite instance.
    """
    def __init__(self, classLabel: str, attributes=None, possibleLabels=None):
        super().__init__(classLabel, attributes)
        if possibleLabels is None:
            possibleLabels = []
        self.possibleClassLabels = possibleLabels

    """
    Accessor for the possible class labels.

    RETURNS
    -------
    list
        Possible class labels of the composite instance.
    """
    def getPossibleClassLabels(self) -> list:
        return self.possibleClassLabels

    """
    Mutator method for possible class labels.

    PARAMETERS
    ----------
    list
        possibleClassLabels Ner value of possible class labels.
    """
    def setPossibleClassLabels(self, possibleClassLabels: list):
        self.possibleClassLabels = possibleClassLabels

    """
    Converts possible class labels to {@link String}.

    RETURNS
    -------
    str
        String representation of possible class labels.
    """
    def __str__(self) -> str:
        result = super().__str__()
        for possibleClassLabel in self.possibleClassLabels:
            result = result + ";" + possibleClassLabel
        return result
from Classification.Instance.Instance import Instance


class KnnInstance(object):
    """
    The constructor that sets the instance and distance value.

    PARAMETERS
    ----------
    instance : Instance
        Instance input.
    distance :float
        Double distance value.
    """

    def __init__(self, instance: Instance, distance: float):
        self.instance = instance
        self.distance = distance

    """
    The toString method returns the concatenation of class label of the instance and the distance value.

    RETURNS
    -------
    str
        The concatenation of class label of the instance and the distance value.
    """

    def __str__(self):
        return self.instance.getClassLabel() + " " + self.distance.__str__()

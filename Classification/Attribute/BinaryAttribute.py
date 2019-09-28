from Classification.Attribute.DiscreteAttribute import DiscreteAttribute


class BinaryAttribute(DiscreteAttribute):

    """
    Constructor for a binary discrete attribute. The attribute can take only two values "True" or "False".

    PARAMETERS
    ----------
    value : str
        Value of the attribute. Can be true or false.
    """
    def __init__(self, value: bool):
        super().__init__(value.__str__())
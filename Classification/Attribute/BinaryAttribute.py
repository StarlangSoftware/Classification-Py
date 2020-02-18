from Classification.Attribute.DiscreteAttribute import DiscreteAttribute


class BinaryAttribute(DiscreteAttribute):

    def __init__(self, value: bool):
        """
        Constructor for a binary discrete attribute. The attribute can take only two values "True" or "False".

        PARAMETERS
        ----------
        value : str
            Value of the attribute. Can be true or false.
        """
        super().__init__(value.__str__())

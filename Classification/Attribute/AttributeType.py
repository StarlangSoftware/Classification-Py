from enum import Enum, auto


class AttributeType(Enum):

    """
    Continuous Attribute
    """
    CONTINUOUS = auto()
    """
    Discrete Attribute
    """
    DISCRETE = auto()
    """
    Binary Attribute
    """
    BINARY = auto()
    """
    Discrete Indexed Attribute is used to store the indices.
    """
    DISCRETE_INDEXED = auto()

from abc import abstractmethod
from Classification.Instance.Instance import Instance


class Model(object):

    """
     An abstract predict method that takes an Instance as an input.

    PARAMETERS
    ----------
    instance : Instance
        Instance to make prediction.

    RETURNS
    -------
    str
        The class label as a String.
    """
    @abstractmethod
    def predict(self, instance: Instance) -> str:
        pass
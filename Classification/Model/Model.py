from abc import abstractmethod
from Classification.Instance.Instance import Instance


class Model(object):

    @abstractmethod
    def predict(self, instance: Instance) -> str:
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
        pass

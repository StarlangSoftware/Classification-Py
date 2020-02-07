from abc import abstractmethod


class Attribute(object):

    @abstractmethod
    def continuousAttributeSize(self) -> int:
        pass

    @abstractmethod
    def continuousAttributes(self) -> list:
        pass

    @abstractmethod
    def getValue(self):
        pass

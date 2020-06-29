import unittest

from Classification.Attribute.AttributeType import AttributeType
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.DataSet.DataSet import DataSet


class ClassifierTest(unittest.TestCase):

    iris: DataSet
    bupa: DataSet
    dermatology: DataSet
    car: DataSet
    tictactoe: DataSet
    nursery: DataSet
    chess: DataSet

    def setUp(self) -> None:
        attributeTypes = 4 * [AttributeType.CONTINUOUS]
        dataDefinition = DataDefinition(attributeTypes)
        self.iris = DataSet(dataDefinition, ",", "../../datasets/iris.data")
        attributeTypes = 6 * [AttributeType.CONTINUOUS]
        dataDefinition = DataDefinition(attributeTypes)
        self.bupa = DataSet(dataDefinition, ",", "../../datasets/bupa.data")
        attributeTypes = 34 * [AttributeType.CONTINUOUS]
        dataDefinition = DataDefinition(attributeTypes)
        self.dermatology = DataSet(dataDefinition, ",", "../../datasets/dermatology.data")
        attributeTypes = 6 * [AttributeType.DISCRETE]
        dataDefinition = DataDefinition(attributeTypes)
        self.car = DataSet(dataDefinition, ",", "../../datasets/car.data")
        attributeTypes = 9 * [AttributeType.DISCRETE]
        dataDefinition = DataDefinition(attributeTypes)
        self.tictactoe = DataSet(dataDefinition, ",", "../../datasets/tictactoe.data")
        attributeTypes = 8 * [AttributeType.DISCRETE]
        dataDefinition = DataDefinition(attributeTypes)
        self.nursery = DataSet(dataDefinition, ",", "../../datasets/nursery.data")
        attributeTypes = []
        for i in range(6):
            if i % 2 == 0:
                attributeTypes.append(AttributeType.DISCRETE)
            else:
                attributeTypes.append(AttributeType.CONTINUOUS)
        dataDefinition = DataDefinition(attributeTypes)
        self.chess = DataSet(dataDefinition, ",", "../../datasets/chess.data")

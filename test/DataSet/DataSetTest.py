import unittest

from Classification.Attribute.AttributeType import AttributeType
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.DataSet.DataSet import DataSet


class DataSetTest(unittest.TestCase):
    iris: DataSet
    bupa: DataSet
    dermatology: DataSet
    car: DataSet
    tictactoe: DataSet
    nursery: DataSet
    chess: DataSet

    def setUp(self):
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

    def test_SampleSize(self):
        self.assertEquals(150, self.iris.sampleSize())
        self.assertEquals(345, self.bupa.sampleSize())
        self.assertEquals(366, self.dermatology.sampleSize())
        self.assertEquals(1728, self.car.sampleSize())
        self.assertEquals(958, self.tictactoe.sampleSize())
        self.assertEquals(12960, self.nursery.sampleSize())
        self.assertEquals(28056, self.chess.sampleSize())

    def test_ClassCount(self):
        self.assertEquals(3, self.iris.classCount())
        self.assertEquals(2, self.bupa.classCount())
        self.assertEquals(6, self.dermatology.classCount())
        self.assertEquals(4, self.car.classCount())
        self.assertEquals(2, self.tictactoe.classCount())
        self.assertEquals(5, self.nursery.classCount())
        self.assertEquals(18, self.chess.classCount())

    def test_GetClasses(self):
        self.assertEquals("Iris-setosa;Iris-versicolor;Iris-virginica", self.iris.getClasses())
        self.assertEquals("1;2", self.bupa.getClasses())
        self.assertEquals("2;1;3;5;4;6", self.dermatology.getClasses())
        self.assertEquals("unacc;acc;vgood;good", self.car.getClasses())
        self.assertEquals("positive;negative", self.tictactoe.getClasses())
        self.assertEquals("recommend;priority;not_recom;very_recom;spec_prior", self.nursery.getClasses())
        self.assertEquals("draw;zero;one;two;three;four;five;six;seven;eight;nine;ten;eleven;twelve;thirteen;fourteen;fifteen;sixteen",
                          self.chess.getClasses())


if __name__ == '__main__':
    unittest.main()

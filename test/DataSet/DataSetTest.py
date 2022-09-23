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
        self.assertEqual(150, self.iris.sampleSize())
        self.assertEqual(345, self.bupa.sampleSize())
        self.assertEqual(366, self.dermatology.sampleSize())
        self.assertEqual(1728, self.car.sampleSize())
        self.assertEqual(958, self.tictactoe.sampleSize())
        self.assertEqual(12960, self.nursery.sampleSize())
        self.assertEqual(28056, self.chess.sampleSize())

    def test_ClassCount(self):
        self.assertEqual(3, self.iris.classCount())
        self.assertEqual(2, self.bupa.classCount())
        self.assertEqual(6, self.dermatology.classCount())
        self.assertEqual(4, self.car.classCount())
        self.assertEqual(2, self.tictactoe.classCount())
        self.assertEqual(5, self.nursery.classCount())
        self.assertEqual(18, self.chess.classCount())

    def test_GetClasses(self):
        self.assertEqual("Iris-setosa;Iris-versicolor;Iris-virginica", self.iris.getClasses())
        self.assertEqual("1;2", self.bupa.getClasses())
        self.assertEqual("2;1;3;5;4;6", self.dermatology.getClasses())
        self.assertEqual("unacc;acc;vgood;good", self.car.getClasses())
        self.assertEqual("positive;negative", self.tictactoe.getClasses())
        self.assertEqual("recommend;priority;not_recom;very_recom;spec_prior", self.nursery.getClasses())
        self.assertEqual("draw;zero;one;two;three;four;five;six;seven;eight;nine;ten;eleven;twelve;thirteen;fourteen;fifteen;sixteen",
                          self.chess.getClasses())


if __name__ == '__main__':
    unittest.main()

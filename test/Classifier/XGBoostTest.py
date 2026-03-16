import unittest

from Classification.Model.Ensemble.XGBoostModel import XGBoostModel
from Classification.Parameter.XGBoostParameter import XGBoostParameter
from test.Classifier.ClassifierTest import ClassifierTest


class XGBoostTest(ClassifierTest):

    def test_Train(self):
        xgboost = XGBoostModel()

        xgboostParameter = XGBoostParameter(
            seed=1,
            n_estimators=50,
            max_depth=4,
            learning_rate=0.3
        )

        # Iris
        xgboost.train(self.iris.getInstanceList(), xgboostParameter)
        self.assertAlmostEqual(
            0.0,
            100 * xgboost.test(self.iris.getInstanceList()).getErrorRate(),
            2
        )

        # Bupa
        xgboost.train(self.bupa.getInstanceList(), xgboostParameter)
        self.assertAlmostEqual(
            0.0,
            100 * xgboost.test(self.bupa.getInstanceList()).getErrorRate(),
            2
        )

        # Dermatology
        xgboost.train(self.dermatology.getInstanceList(), xgboostParameter)
        self.assertAlmostEqual(
            0.0,
            100 * xgboost.test(self.dermatology.getInstanceList()).getErrorRate(),
            2
        )

        # Car
        xgboost.train(self.car.getInstanceList(), xgboostParameter)
        self.assertAlmostEqual(
            0.0,
            100 * xgboost.test(self.car.getInstanceList()).getErrorRate(),
            2
        )

        # TicTacToe
        xgboost.train(self.tictactoe.getInstanceList(), xgboostParameter)
        self.assertAlmostEqual(
            0.0,
            100 * xgboost.test(self.tictactoe.getInstanceList()).getErrorRate(),
            2
        )


if __name__ == '__main__':
    unittest.main()
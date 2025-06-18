import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dsl import parser

class TestParser(unittest.TestCase):
    def test_parse_train_model(self):
        text = (
            "TRAIN MODEL fraud_detector USING logistic_regression(regularization=0.01) FROM transactions "
            "PREDICT is_fraud WITH FEATURES(amount, merchant_type)"
        )
        model = parser.parse(text)
        self.assertEqual(model.name, "fraud_detector")
        self.assertEqual(model.algorithm, "logistic_regression")
        self.assertEqual(model.source, "transactions")
        self.assertEqual(model.target, "is_fraud")
        self.assertEqual(model.features, ["amount", "merchant_type"])
        sql = parser.compile_sql(model)
        self.assertIn("ml_train_model", sql)

    def test_parse_train_model_no_params(self):
        text = (
            "TRAIN MODEL x USING logistic_regression FROM data "
            "PREDICT y WITH FEATURES(a)"
        )
        model = parser.parse(text)
        self.assertEqual(model.name, "x")
        self.assertEqual(model.algorithm, "logistic_regression")
        self.assertEqual(model.params, [])
        self.assertEqual(model.source, "data")
        self.assertEqual(model.target, "y")
        self.assertEqual(model.features, ["a"])

if __name__ == "__main__":
    unittest.main()

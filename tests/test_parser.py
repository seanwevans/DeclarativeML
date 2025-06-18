import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dsl import parser  # noqa: E402


class TestParser(unittest.TestCase):
    def test_parse_train_model(self):
        text = (
            "TRAIN MODEL fraud_detector USING logistic_regression("
            "regularization=0.01) FROM transactions "
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
            "TRAIN MODEL simple_model USING decision_tree FROM training_data "
            "PREDICT outcome WITH FEATURES(a, b)"
        )
        model = parser.parse(text)
        self.assertEqual(model.name, "simple_model")
        self.assertEqual(model.algorithm, "decision_tree")
        self.assertEqual(model.params, [])
        self.assertEqual(model.source, "training_data")
        self.assertEqual(model.target, "outcome")
        self.assertEqual(model.features, ["a", "b"])


if __name__ == "__main__":
    unittest.main()

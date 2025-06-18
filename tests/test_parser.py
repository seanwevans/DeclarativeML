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

    def test_parse_train_model_with_options(self):
        text = (
            "TRAIN MODEL m USING alg() FROM data PREDICT y WITH FEATURES(f1, f2) "
            "SPLIT DATA training=0.7, validation=0.2, test=0.1 "
            "VALIDATE USING cv(folds=5) OPTIMIZE FOR accuracy "
            "STOP WHEN accuracy > 0.9"
        )
        model = parser.parse(text)
        self.assertIsNotNone(model.split)
        self.assertAlmostEqual(model.split.ratios["training"], 0.7)
        self.assertIsNotNone(model.validate)
        self.assertEqual(model.validate.method, "cv")
        self.assertEqual(model.optimize_metric, "accuracy")
        self.assertEqual(model.stop_condition, "accuracy > 0.9")

if __name__ == "__main__":
    unittest.main()

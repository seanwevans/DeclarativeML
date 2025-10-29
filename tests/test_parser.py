import json
import re
import unittest
from typing import cast

from hypothesis import given
from hypothesis import strategies as st
from lark.exceptions import LarkError

from dsl import parser


class TestParser(unittest.TestCase):
    def test_parse_train_model(self):
        text = (
            "TRAIN MODEL fraud_detector USING logistic_regression("
            "regularization=0.01) FROM transactions "
            "PREDICT is_fraud WITH FEATURES(amount, merchant_type)"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertEqual(model.name, "fraud_detector")
        self.assertEqual(model.algorithm, "logistic_regression")
        self.assertEqual(model.source, "transactions")
        self.assertEqual(model.target, "is_fraud")
        self.assertEqual(model.features, ["amount", "merchant_type"])
        self.assertTrue(model.source_is_identifier)
        sql = parser.compile_sql(model)
        self.assertIn("ml_train_model", sql)

    def test_parse_train_model_no_params(self):
        text = (
            "TRAIN MODEL simple_model USING decision_tree FROM training_data "
            "PREDICT outcome WITH FEATURES(a, b)"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertEqual(model.name, "simple_model")
        self.assertEqual(model.algorithm, "decision_tree")
        self.assertEqual(model.params, [])
        self.assertEqual(model.source, "training_data")
        self.assertEqual(model.target, "outcome")
        self.assertEqual(model.features, ["a", "b"])
        self.assertTrue(model.source_is_identifier)

    def test_parse_train_model_join_source(self):
        text = (
            "TRAIN MODEL joined USING alg FROM transactions JOIN merchants ON "
            "transactions.merchant_id = merchants.id PREDICT y WITH FEATURES(a)"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertEqual(
            model.source,
            "transactions JOIN merchants ON transactions.merchant_id = merchants.id",
        )
        self.assertFalse(model.source_is_identifier)
        sql_str = parser.compile_sql(model)
        self.assertIn("JOIN merchants", sql_str)

    def test_parse_train_model_filtered_source(self):
        text = (
            "TRAIN MODEL filtered USING alg FROM (SELECT * FROM base WHERE active = TRUE) sub "
            "PREDICT y WITH FEATURES(a)"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertEqual(
            model.source,
            "(SELECT * FROM base WHERE active = TRUE) sub",
        )
        self.assertFalse(model.source_is_identifier)
        sql_str = parser.compile_sql(model)
        self.assertIn("FROM (SELECT * FROM base WHERE active = TRUE) sub", sql_str)

    def test_parse_train_model_with_options(self):
        text = (
            "TRAIN MODEL m USING alg() FROM data PREDICT y "
            "WITH FEATURES(f1, f2) "
            "SPLIT DATA training=0.7, validation=0.2, test=0.1 "
            "VALIDATE USING cv(folds=5) OPTIMIZE FOR accuracy "
            "STOP WHEN accuracy > 0.9"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertIsNotNone(model.split)
        self.assertAlmostEqual(model.split.ratios["training"], 0.7)
        self.assertIsNotNone(model.validate)
        self.assertEqual(model.validate.method, "cv")
        self.assertEqual(model.optimize_metric, "accuracy")
        self.assertEqual(model.stop_condition, "accuracy > 0.9")

    def test_feature_list_with_expressions(self):
        text = (
            "TRAIN MODEL m USING alg() FROM data PREDICT y WITH FEATURES("
            "amount, DERIVED(amount * exchange_rate), "
            "TRANSFORM(scale(log(amount + 1))))"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertEqual(
            model.features,
            [
                "amount",
                "DERIVED(amount * exchange_rate)",
                "TRANSFORM(scale(log(amount + 1)))",
            ],
        )

    def test_feature_string_with_embedded_quotes(self):
        text = (
            'TRAIN MODEL quoted USING alg FROM source '
            'PREDICT target WITH FEATURES("text \\"with\\" quotes")'
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertEqual(model.features, ['"text ""with"" quotes"'])

        sql = parser.compile_sql(model)
        match = re.search(r"feature_columns := ARRAY\[\s*(?:E)?'([^']*)'\]", sql)
        self.assertIsNotNone(match)
        literal_body = match.group(1)
        decoded = literal_body.encode("utf-8").decode("unicode_escape")
        final_literal = decoded.replace('\\"', '"')
        self.assertEqual('"text ""with"" quotes"', final_literal)

    def test_compile_sql_with_feature_expressions(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="source_table",
            target="target_col",
            features=[
                "amount",
                "DERIVED(amount * exchange_rate)",
                "TRANSFORM(scale(log(amount + 1)))",
            ],
        )
        sql_str = parser.compile_sql(model)
        self.assertIn('"amount"', sql_str)
        self.assertIn("DERIVED(amount * exchange_rate)", sql_str)
        self.assertIn("TRANSFORM(scale(log(amount + 1)))", sql_str)

    def test_invalid_syntax_raises(self):
        with self.assertRaises(LarkError):
            parser.parse("TRAIN MODEL bad USING algo FROM tbl")

    def test_missing_features_clause(self):
        text = "TRAIN MODEL m USING a FROM t PREDICT y"
        with self.assertRaises(LarkError):
            parser.parse(text)

    def test_empty_feature_list(self):
        text = "TRAIN MODEL m USING a FROM t PREDICT y WITH FEATURES()"
        with self.assertRaises(LarkError):
            parser.parse(text)

    def test_algorithm_param_types(self):
        text = (
            'TRAIN MODEL m USING alg(num=1, rate=0.5, name="x") FROM t '
            "PREDICT y WITH FEATURES(a)"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertEqual(
            model.params,
            [
                ("num", 1),
                ("rate", 0.5),
                ("name", "x"),
            ],
        )

    def test_negative_param_values(self):
        text = (
            "TRAIN MODEL m USING alg(alpha=-0.1, depth=-5) FROM t "
            "PREDICT y WITH FEATURES(a)"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertEqual(model.params, [("alpha", -0.1), ("depth", -5)])

    def test_algorithm_param_list_and_dict_literals(self):
        text = (
            "TRAIN MODEL m USING alg("
            "layers=[64, 128, 256], "
            "config={mode: fast, thresholds: [0.1, 0.2]}"
            ") FROM t PREDICT y WITH FEATURES(a)"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertEqual(
            model.params,
            [
                ("layers", [64, 128, 256]),
                (
                    "config",
                    {
                        "mode": "fast",
                        "thresholds": [0.1, 0.2],
                    },
                ),
            ],
        )
        sql = parser.compile_sql(model)
        match = re.search(r"algorithm_params := '([^']*)'", sql)
        self.assertIsNotNone(match)
        params_json = match.group(1)
        self.assertEqual(
            json.loads(params_json),
            {
                "layers": [64, 128, 256],
                "config": {"mode": "fast", "thresholds": [0.1, 0.2]},
            },
        )

    def test_balance_clause(self):
        text = (
            "TRAIN MODEL m USING alg() FROM t PREDICT y WITH FEATURES(a) "
            "BALANCE CLASSES BY oversampling"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertEqual(model.balance_method, "oversampling")

    def test_checkpoint_clause(self):
        text = (
            "TRAIN MODEL m USING alg() FROM t PREDICT y WITH FEATURES(a) "
            "SAVE CHECKPOINTS EVERY 10 epochs"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertIsNotNone(model.checkpoint)
        assert model.checkpoint is not None
        self.assertEqual(model.checkpoint.interval, 10)
        self.assertEqual(model.checkpoint.unit, "epochs")

    def test_parse_compute(self):
        text = (
            "COMPUTE add_vectors FROM table(foo, bar) INTO column(baz) "
            "USING vector_add BLOCK 256 GRID auto"
        )
        stmt = cast(parser.ComputeKernel, parser.parse(text))
        self.assertIsInstance(stmt, parser.ComputeKernel)
        self.assertEqual(stmt.name, "add_vectors")
        self.assertEqual(stmt.inputs, ["foo", "bar"])
        self.assertEqual(stmt.output, "baz")
        self.assertEqual(stmt.kernel, "vector_add")
        self.assertEqual(stmt.options["BLOCK"], 256)
        self.assertEqual(stmt.options["GRID"], "auto")

    def test_parse_compute_every(self):
        text = "COMPUTE scan_peptides EVERY 1000 TICKS USING immune_scan SHARED 1K"
        stmt = cast(parser.ComputeKernel, parser.parse(text))
        self.assertEqual(stmt.schedule_ticks, 1000)
        self.assertEqual(stmt.kernel, "immune_scan")
        self.assertEqual(stmt.options["SHARED"], "1K")

    def test_parse_compute_invalid_clause(self):
        text = "COMPUTE bad_job USING some_kernel EXTRA"
        with self.assertRaises(LarkError):
            parser.parse(text)

    def test_compute_stmt_unexpected_part(self):
        transformer = parser.TreeToModel()
        with self.assertRaises(ValueError) as ctx:
            transformer.compute_stmt("bad_job", "kernel", 123)
        self.assertIn("Unexpected compute clause part", str(ctx.exception))

    def test_data_split_sum_validation_passes(self):
        text = (
            "TRAIN MODEL m USING alg() FROM t PREDICT y WITH FEATURES(a, b) "
            "SPLIT DATA train=0.8, test=0.2"
        )
        model = parser.parse(text)
        self.assertIsNotNone(model.split)
        self.assertAlmostEqual(sum(model.split.ratios.values()), 1.0)

    def test_data_split_sum_validation_fails(self):
        text = (
            "TRAIN MODEL m USING alg() FROM t PREDICT y WITH FEATURES(a, b) "
            "SPLIT DATA train=0.6, test=0.3"
        )
        with self.assertRaises(ValueError):
            parser.parse(text)


    def test_compute_missing_kernel(self):
        text = "COMPUTE add_vectors FROM table(foo) INTO column(bar)"
        with self.assertRaises(LarkError):
            parser.parse(text)

    def test_compile_sql_escapes_identifiers(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="weird;table",
            target="tar;get",
            features=["fe;ature"],
        )
        sql_str = parser.compile_sql(model)
        self.assertIn('"weird;table"', sql_str)
        self.assertIn('"fe;ature"', sql_str)
        self.assertIn('"tar;get"', sql_str)

    def test_compile_sql_includes_checkpoint(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="data",
            target="target",
            features=["feature"],
            checkpoint=parser.CheckpointOption(interval=5, unit="epochs"),
        )
        sql_str = parser.compile_sql(model)
        self.assertIn("checkpoint_schedule :=", sql_str)
        self.assertIn('"interval": 5', sql_str)
        self.assertIn('"unit": "epochs"', sql_str)

    def test_compile_sql_escapes_compute_identifiers(self):
        stmt = parser.ComputeKernel(
            name="name;drop",
            inputs=["in;put"],
            output="out;put",
            schedule_ticks=None,
            kernel="ker;nel",
            options=None,
        )
        sql_str = parser.compile_sql(stmt)
        self.assertIn("'name;drop'", sql_str)
        self.assertIn("'ker;nel'", sql_str)
        self.assertIn("'in;put'", sql_str)
        self.assertIn("'out;put'", sql_str)


@given(
    model_name=st.text(
        min_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
    ),
    algorithm=st.text(
        min_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
    ),
    source=st.text(
        min_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
    ),
    target=st.text(
        min_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
    ),
    feature=st.text(
        min_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
    ),
)
def test_property_based_parse(model_name, algorithm, source, target, feature):
    text = (
        f"TRAIN MODEL {model_name} USING {algorithm} FROM {source} "
        f"PREDICT {target} WITH FEATURES({feature})"
    )
    model = cast(parser.TrainModel, parser.parse(text))
    assert model.name == model_name
    assert model.algorithm == algorithm


if __name__ == "__main__":
    unittest.main()

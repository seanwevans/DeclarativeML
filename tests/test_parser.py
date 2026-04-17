import json
import re
import unittest
from typing import cast

from hypothesis import given
from hypothesis import strategies as st
from lark.exceptions import LarkError

from dsl import parser


def _extract_named_arg(sql_text: str, arg_name: str) -> str:
    escaped_name = re.escape(arg_name)
    match = re.search(
        rf"(?<![A-Za-z0-9_]){escaped_name}(?![A-Za-z0-9_])\s*:=\s*"
        rf"(?P<value>ARRAY\[(?:.|\n)*?\]|'(?:''|[^'])*'|-?\d+(?:\.\d+)?)\s*(?:,|\))",
        sql_text,
    )
    if not match:
        raise AssertionError(f"Argument '{arg_name}' not found in SQL: {sql_text}")
    return match.group("value")


def _decode_sql_string_literal(value: str) -> str:
    if len(value) < 2 or value[0] != "'" or value[-1] != "'":
        raise AssertionError(f"Expected SQL string literal, got: {value}")
    return value[1:-1].replace("''", "'")


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

    def test_parse_train_model_source_identifier_boundaries(self):
        cases = [
            ("transactions", True),
            ("analytics.transactions", False),
            ('"Transactions"', False),
            ("transactions JOIN merchants ON transactions.id = merchants.id", False),
            ("(SELECT * FROM transactions) t", False),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                text = (
                    f"TRAIN MODEL m USING alg FROM {source} "
                    "PREDICT y WITH FEATURES(a)"
                )
                model = cast(parser.TrainModel, parser.parse(text))
                self.assertEqual(model.source, source)
                self.assertEqual(model.source_is_identifier, expected)

    def test_compile_sql_uses_identifier_mode_for_simple_source(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="transactions",
            target="y",
            features=["a"],
            source_is_identifier=True,
        )
        sql_str = parser.compile_sql(model)
        self.assertIn('FROM "transactions"', sql_str)

    def test_compile_sql_uses_fragment_mode_for_dotted_source(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="analytics.transactions",
            target="y",
            features=["a"],
            source_is_identifier=False,
        )
        sql_str = parser.compile_sql(model)
        self.assertIn("FROM analytics.transactions", sql_str)
        self.assertNotIn('FROM "analytics.transactions"', sql_str)

    def test_compile_sql_uses_fragment_mode_for_quoted_source(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source='"Transactions"',
            target="y",
            features=["a"],
            source_is_identifier=False,
        )
        sql_str = parser.compile_sql(model)
        self.assertIn('FROM "Transactions"', sql_str)

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
        self.assertNotIn(
            'FROM "transactions JOIN merchants ON transactions.merchant_id = merchants.id"',
            sql_str,
        )

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
        self.assertNotIn(
            'FROM "(SELECT * FROM base WHERE active = TRUE) sub"',
            sql_str,
        )

    def test_parse_train_model_source_with_predict_in_string_literal(self):
        text = (
            "TRAIN MODEL filtered USING alg FROM transactions t "
            "WHERE t.note = 'PREDICT' PREDICT y WITH FEATURES(a)"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertEqual(
            model.source,
            "transactions t WHERE t.note = 'PREDICT'",
        )
        self.assertEqual(model.target, "y")
        self.assertFalse(model.source_is_identifier)

    def test_parse_train_model_source_with_predict_in_alias(self):
        text = (
            "TRAIN MODEL filtered USING alg FROM (SELECT * FROM transactions) predict_alias "
            "PREDICT y WITH FEATURES(a)"
        )
        model = cast(parser.TrainModel, parser.parse(text))
        self.assertEqual(
            model.source,
            "(SELECT * FROM transactions) predict_alias",
        )
        self.assertEqual(model.target, "y")
        self.assertFalse(model.source_is_identifier)

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
        self.assertEqual(model.features, ['"text \\"with\\" quotes"'])

        sql = parser.compile_sql(model)
        match = re.search(r"feature_columns := ARRAY\[\s*(?:E)?'([^']*)'\]", sql)
        self.assertIsNotNone(match)
        literal_body = match.group(1)
        decoded = literal_body.encode("utf-8").decode("unicode_escape")
        self.assertEqual('"text \\"with\\" quotes"', decoded)

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

    def test_compile_sql_with_dotted_identifier(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="source_table",
            target="target_col",
            features=["amount", "customer.age"],
        )
        sql_str = parser.compile_sql(model)
        match = re.search(r"training_data := '([^']*)'", sql_str)
        self.assertIsNotNone(match)
        training_query = match.group(1)
        self.assertIn('"amount"', training_query)
        self.assertIn('"customer"."age"', training_query)

    def test_compile_sql_with_operator_expression(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="source_table",
            target="target_col",
            features=["amount", "amount + tax"],
        )
        sql_str = parser.compile_sql(model)
        match = re.search(r"training_data := '([^']*)'", sql_str)
        self.assertIsNotNone(match)
        training_query = match.group(1)
        self.assertIn('"amount"', training_query)
        self.assertIn('("amount" + "tax")', training_query)
        self.assertNotIn('"amount + tax"', training_query)

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
        params_json = _decode_sql_string_literal(_extract_named_arg(sql, "algorithm_params"))
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

    def test_parse_compute_valid_block_and_shared_edges(self):
        text = "COMPUTE scan_peptides USING immune_scan BLOCK 1 SHARED 0 GRID auto"
        stmt = cast(parser.ComputeKernel, parser.parse(text))
        self.assertEqual(stmt.options["BLOCK"], 1)
        self.assertEqual(stmt.options["SHARED"], "0")
        self.assertEqual(stmt.options["GRID"], "auto")

    def test_parse_compute_every_fractional_ticks(self):
        text = "COMPUTE scan_peptides EVERY 10.5 TICKS USING immune_scan"
        with self.assertRaises(ValueError):
            parser.parse(text)

    def test_parse_compute_every_non_positive_ticks(self):
        text = "COMPUTE scan_peptides EVERY 0 TICKS USING immune_scan"
        with self.assertRaises(ValueError):
            parser.parse(text)

    def test_parse_compute_invalid_clause(self):
        text = "COMPUTE bad_job USING some_kernel EXTRA"
        with self.assertRaises(LarkError):
            parser.parse(text)

    def test_parse_compute_invalid_block_values(self):
        with self.assertRaisesRegex(ValueError, "block size must be a positive integer"):
            parser.parse("COMPUTE bad_job USING some_kernel BLOCK 0")

        with self.assertRaisesRegex(ValueError, "block size must be a positive integer"):
            parser.parse("COMPUTE bad_job USING some_kernel BLOCK -2")

        with self.assertRaisesRegex(ValueError, "block size must be a positive integer"):
            parser.parse("COMPUTE bad_job USING some_kernel BLOCK 32.5")

    def test_parse_compute_invalid_shared_values(self):
        with self.assertRaisesRegex(
            ValueError,
            "shared memory size must be a non-negative integer optionally suffixed with K, M, or G",
        ):
            parser.parse("COMPUTE bad_job USING some_kernel SHARED -1")

        with self.assertRaisesRegex(
            ValueError,
            "shared memory size must be a non-negative integer optionally suffixed with K, M, or G",
        ):
            parser.parse("COMPUTE bad_job USING some_kernel SHARED 1.5K")

        with self.assertRaisesRegex(
            ValueError,
            "shared memory size must be a non-negative integer optionally suffixed with K, M, or G",
        ):
            parser.parse("COMPUTE bad_job USING some_kernel SHARED 2KB")

    def test_parse_compute_invalid_grid_values(self):
        with self.assertRaisesRegex(ValueError, "grid value must be one of: auto"):
            parser.parse("COMPUTE bad_job USING some_kernel GRID manual")

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
        with self.assertRaises(ValueError):
            parser.compile_sql(model)

    def test_compile_sql_quotes_single_table_with_punctuation(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="user-events",
            target="target",
            features=["feature"],
            source_is_identifier=True,
        )
        sql_str = parser.compile_sql(model)
        self.assertIn('FROM "user-events"', sql_str)

    def test_compile_sql_blocks_unsafe_source_semicolon(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="transactions; DROP TABLE users",
            target="target",
            features=["feature"],
            source_is_identifier=False,
        )
        with self.assertRaises(ValueError):
            parser.compile_sql(model)

    def test_compile_sql_blocks_unsafe_source_keywords(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="transactions WHERE 1=1 COMMIT",
            target="target",
            features=["feature"],
            source_is_identifier=False,
        )
        with self.assertRaises(ValueError):
            parser.compile_sql(model)

    def test_compile_sql_allows_safe_join_source(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="transactions t JOIN merchants m ON t.merchant_id = m.id WHERE t.amount > 0",
            target="target",
            features=["t.amount", "m.category"],
            source_is_identifier=False,
        )
        sql_str = parser.compile_sql(model)
        self.assertIn("JOIN merchants m ON t.merchant_id = m.id", sql_str)
        self.assertIn('"t"."amount"', sql_str)

    def test_compile_sql_allows_safe_parenthesized_subquery(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="(SELECT * FROM transactions WHERE amount > 10) tx",
            target="target",
            features=["amount * 2", "sqrt(amount + 1)"],
            source_is_identifier=False,
        )
        sql_str = parser.compile_sql(model)
        self.assertIn("FROM (SELECT * FROM transactions WHERE amount > 10) tx", sql_str)
        self.assertIn('("amount" * 2)', sql_str)
        self.assertIn('"sqrt"(("amount" + 1))', sql_str)

    def test_compile_sql_blocks_unsafe_feature_expression(self):
        model = parser.TrainModel(
            name="m",
            algorithm="alg",
            params=[],
            source="source_table",
            target="target_col",
            features=["amount", "amount; DROP TABLE users"],
        )
        with self.assertRaises(ValueError):
            parser.compile_sql(model)

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
        checkpoint_payload = json.loads(
            _decode_sql_string_literal(_extract_named_arg(sql_str, "checkpoint_schedule"))
        )
        self.assertEqual(checkpoint_payload, {"interval": 5, "unit": "epochs"})

    def test_compile_sql_train_structure_with_multiple_options(self):
        model = parser.TrainModel(
            name="fraud_v2",
            algorithm="xgboost",
            params=[("max_depth", 6), ("learning_rate", 0.1)],
            source="transactions",
            target="is_fraud",
            features=["amount", "merchant_type"],
            split=parser.DataSplit({"training": 0.7, "validation": 0.2, "test": 0.1}),
            validate=parser.ValidationOption(method="cv", params=[("folds", 5)]),
            optimize_metric="f1_score",
            checkpoint=parser.CheckpointOption(interval=10, unit="epochs"),
        )
        sql_str = parser.compile_sql(model)

        # Smoke checks
        self.assertIn("ml_train_model", sql_str)
        self.assertIn("model_name :=", sql_str)
        self.assertIn("training_data :=", sql_str)

        # Structure checks
        self.assertEqual(
            _decode_sql_string_literal(_extract_named_arg(sql_str, "model_name")),
            "fraud_v2",
        )
        self.assertEqual(
            _decode_sql_string_literal(_extract_named_arg(sql_str, "algorithm")),
            "xgboost",
        )

        training_data = _decode_sql_string_literal(_extract_named_arg(sql_str, "training_data"))
        self.assertIn('FROM "transactions"', training_data)
        self.assertIn('"amount"', training_data)
        self.assertIn('"merchant_type"', training_data)
        self.assertIn('"is_fraud"', training_data)

        self.assertEqual(
            json.loads(
                _decode_sql_string_literal(_extract_named_arg(sql_str, "algorithm_params"))
            ),
            {"max_depth": 6, "learning_rate": 0.1},
        )
        self.assertEqual(
            json.loads(_decode_sql_string_literal(_extract_named_arg(sql_str, "data_split"))),
            {"training": 0.7, "validation": 0.2, "test": 0.1},
        )
        self.assertEqual(
            _decode_sql_string_literal(_extract_named_arg(sql_str, "validate_method")),
            "cv",
        )
        self.assertEqual(
            json.loads(
                _decode_sql_string_literal(_extract_named_arg(sql_str, "validate_params"))
            ),
            {"folds": 5},
        )
        self.assertEqual(
            _decode_sql_string_literal(_extract_named_arg(sql_str, "optimize_metric")),
            "f1_score",
        )
        self.assertEqual(
            json.loads(
                _decode_sql_string_literal(_extract_named_arg(sql_str, "checkpoint_schedule"))
            ),
            {"interval": 10, "unit": "epochs"},
        )

    def test_compile_sql_compute_structure_with_schedule_and_options(self):
        stmt = parser.ComputeKernel(
            name="scan_peptides",
            kernel="immune_scan",
            inputs=["signal_a", "signal_b"],
            output="risk_score",
            schedule_ticks=1000,
            options={"BLOCK": 256, "GRID": "auto", "SHARED": "1K"},
        )
        sql_str = parser.compile_sql(stmt)

        # Smoke checks
        self.assertIn("ml_register_compute", sql_str)
        self.assertIn("schedule_ticks :=", sql_str)
        self.assertIn("options :=", sql_str)

        # Structure checks
        self.assertEqual(
            _decode_sql_string_literal(_extract_named_arg(sql_str, "kernel_name")),
            "immune_scan",
        )
        self.assertEqual(
            _decode_sql_string_literal(_extract_named_arg(sql_str, "name")),
            "scan_peptides",
        )
        self.assertEqual(_extract_named_arg(sql_str, "schedule_ticks"), "1000")
        self.assertEqual(
            json.loads(_decode_sql_string_literal(_extract_named_arg(sql_str, "options"))),
            {"BLOCK": 256, "GRID": "auto", "SHARED": "1K"},
        )

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

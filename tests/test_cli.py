import json
import os
import re
import subprocess
import sys
import tempfile
import unittest


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


class TestCLI(unittest.TestCase):
    def test_cli_stdin(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        dsl_text = (
            "TRAIN MODEL cli_model USING decision_tree FROM orders JOIN customers ON "
            "orders.customer_id = customers.id PREDICT label WITH FEATURES(x, y)"
        )
        result = subprocess.run(
            [sys.executable, "-m", "dsl.cli"],
            input=dsl_text.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repo_root,
            check=True,
        )
        output = result.stdout.decode()
        self.assertIn("ml_train_model", output)

    def test_cli_compute(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        dsl_text = (
            "COMPUTE add_vectors FROM table(a, b) INTO column(c) "
            "USING vector_add BLOCK 128"
        )
        result = subprocess.run(
            [sys.executable, "-m", "dsl.cli"],
            input=dsl_text.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repo_root,
            check=True,
        )
        output = result.stdout.decode()
        self.assertIn("ml_register_compute", output)

    def test_cli_file(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        dsl_text = (
            "TRAIN MODEL file_model USING decision_tree FROM data "
            "PREDICT label WITH FEATURES(x, y)"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".dsl", delete=False) as tmp:
            tmp.write(dsl_text)
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                [sys.executable, "-m", "dsl.cli", tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=repo_root,
                check=True,
            )
        finally:
            os.remove(tmp_path)
        output = result.stdout.decode()
        self.assertIn("ml_train_model", output)

    def test_cli_invalid_input(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        bad_dsl = "TRAIN MODEL"
        result = subprocess.run(
            [sys.executable, "-m", "dsl.cli"],
            input=bad_dsl.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repo_root,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_cli_missing_file(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        missing_path = os.path.join(repo_root, "does_not_exist.dsl")
        result = subprocess.run(
            [sys.executable, "-m", "dsl.cli", missing_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repo_root,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to read source file", result.stderr.decode())

    def test_cli_outputs_nested_params(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        dsl_text = (
            "TRAIN MODEL nested USING algo("
            "layers=[32, 16], config={mode: fast, thresholds: [0.1, 0.2]}"
            ") FROM data PREDICT label WITH FEATURES(x)"
        )
        result = subprocess.run(
            [sys.executable, "-m", "dsl.cli"],
            input=dsl_text.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repo_root,
            check=True,
        )
        output = result.stdout.decode()
        self.assertIn("ml_train_model", output)
        params = json.loads(
            _decode_sql_string_literal(_extract_named_arg(output, "algorithm_params"))
        )
        self.assertEqual(
            params,
            {
                "layers": [32, 16],
                "config": {"mode": "fast", "thresholds": [0.1, 0.2]},
            },
        )

    def test_cli_train_with_split_validate_optimize_and_checkpoint(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        dsl_text = (
            "TRAIN MODEL cli_contract USING decision_tree(max_depth=8) FROM train_data "
            "PREDICT label WITH FEATURES(x, y) "
            "SPLIT DATA training=0.7, validation=0.2, test=0.1 "
            "VALIDATE USING cv(folds=4) OPTIMIZE FOR accuracy "
            "SAVE CHECKPOINTS EVERY 5 epochs"
        )
        result = subprocess.run(
            [sys.executable, "-m", "dsl.cli"],
            input=dsl_text.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repo_root,
            check=True,
        )
        output = result.stdout.decode()

        # Smoke checks
        self.assertIn("ml_train_model", output)
        self.assertIn("model_name :=", output)
        self.assertIn("training_data :=", output)

        # Structure checks
        self.assertEqual(
            _decode_sql_string_literal(_extract_named_arg(output, "model_name")),
            "cli_contract",
        )
        self.assertEqual(
            _decode_sql_string_literal(_extract_named_arg(output, "training_data"))
            .split(" FROM ")[-1],
            '"train_data"',
        )
        self.assertEqual(
            json.loads(
                _decode_sql_string_literal(_extract_named_arg(output, "algorithm_params"))
            ),
            {"max_depth": 8},
        )
        self.assertEqual(
            json.loads(_decode_sql_string_literal(_extract_named_arg(output, "data_split"))),
            {"training": 0.7, "validation": 0.2, "test": 0.1},
        )
        self.assertEqual(
            _decode_sql_string_literal(_extract_named_arg(output, "validate_method")),
            "cv",
        )
        self.assertEqual(
            json.loads(
                _decode_sql_string_literal(_extract_named_arg(output, "validate_params"))
            ),
            {"folds": 4},
        )
        self.assertEqual(
            _decode_sql_string_literal(_extract_named_arg(output, "optimize_metric")),
            "accuracy",
        )
        self.assertEqual(
            json.loads(
                _decode_sql_string_literal(_extract_named_arg(output, "checkpoint_schedule"))
            ),
            {"interval": 5, "unit": "epochs"},
        )

    def test_cli_compute_with_schedule_and_options_contract(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        dsl_text = (
            "COMPUTE scan_peptides EVERY 1000 TICKS USING immune_scan "
            "BLOCK 256 GRID auto SHARED 1K"
        )
        result = subprocess.run(
            [sys.executable, "-m", "dsl.cli"],
            input=dsl_text.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repo_root,
            check=True,
        )
        output = result.stdout.decode()

        # Smoke checks
        self.assertIn("ml_register_compute", output)
        self.assertIn("schedule_ticks :=", output)
        self.assertIn("options :=", output)

        # Structure checks
        self.assertEqual(
            _decode_sql_string_literal(_extract_named_arg(output, "kernel_name")),
            "immune_scan",
        )
        self.assertEqual(
            _decode_sql_string_literal(_extract_named_arg(output, "name")),
            "scan_peptides",
        )
        self.assertEqual(_extract_named_arg(output, "schedule_ticks"), "1000")
        self.assertEqual(
            json.loads(_decode_sql_string_literal(_extract_named_arg(output, "options"))),
            {"BLOCK": 256, "GRID": "auto", "SHARED": "1K"},
        )



if __name__ == "__main__":
    unittest.main()

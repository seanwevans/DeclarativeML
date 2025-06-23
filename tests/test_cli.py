import os
import subprocess
import sys
import tempfile
import unittest


class TestCLI(unittest.TestCase):
    def test_cli_stdin(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        dsl_text = (
            "TRAIN MODEL cli_model USING decision_tree FROM data "
            "PREDICT label WITH FEATURES(x, y)"
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


if __name__ == "__main__":
    unittest.main()

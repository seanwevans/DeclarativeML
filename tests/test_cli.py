import os
import subprocess
import sys
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


if __name__ == "__main__":
    unittest.main()

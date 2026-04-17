from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_import_dsl_is_side_effect_minimal() -> None:
    project_root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import dsl; "
                "print(json.dumps({"
                "'has_cli_module': 'dsl.cli' in sys.modules, "
                "'all': dsl.__all__"
                "}))"
            ),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout.strip())
    assert payload["has_cli_module"] is False
    assert payload["all"] == ["TrainModel", "ComputeKernel", "parse", "compile_sql"]

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_pip_install_includes_psycopg(tmp_path):
    project_root = Path(__file__).resolve().parent.parent
    install_dir = tmp_path / "install"
    install_dir.mkdir()

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            project_root.as_posix(),
            "--target",
            install_dir.as_posix(),
        ],
        cwd=project_root,
    )

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        install_dir.as_posix()
        if not existing_pythonpath
        else os.pathsep.join([install_dir.as_posix(), existing_pythonpath])
    )

    subprocess.check_call(
        [sys.executable, "-c", "import psycopg"],
        env=env,
    )

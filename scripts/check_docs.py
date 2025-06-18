import os
import sys

required_files = [
    "README.md",
    "AGENTS.md",
    "DISTRIBUTED.md",
    "DSL.md",
    "EXTENSIONS.md",
]


def main() -> None:
    """Validate the presence of required documentation files."""
    missing = [f for f in required_files if not os.path.isfile(f)]
    if missing:
        print("Missing required documentation files:", ", ".join(missing))
        sys.exit(1)
    else:
        print("All required documentation files are present.")


if __name__ == "__main__":
    main()

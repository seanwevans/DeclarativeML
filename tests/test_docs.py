import os
import re

DOC_FILES = [
    "README.md",
    "AGENTS.md",
    "DSL.md",
    "EXTENSIONS.md",
    "DISTRIBUTED.md",
]


def test_doc_files_exist():
    """Ensure all required documentation files are present."""
    missing = [f for f in DOC_FILES if not os.path.exists(f)]
    assert not missing, f"Missing documentation file(s): {', '.join(missing)}"


def test_markdown_headings():
    pattern = re.compile(r"^# .+", re.MULTILINE)
    for f in DOC_FILES:
        with open(f, "r", encoding="utf-8") as fh:
            content = fh.read()
        assert pattern.search(content), f"{f} missing top-level heading"


def test_readme_links():
    with open("README.md", "r", encoding="utf-8") as fh:
        readme = fh.read()
    # Check that README links to each doc file
    for f in DOC_FILES:
        if f == "README.md":
            continue
        assert f"`{f}`" in readme, f"README.md missing reference to {f}"

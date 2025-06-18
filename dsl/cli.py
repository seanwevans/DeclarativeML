import argparse
import sys

from .parser import parse, compile_sql


def main(argv: list[str] | None = None) -> int:
    """Compile DeclarativeML DSL to SQL."""
    parser = argparse.ArgumentParser(
        description="Compile DeclarativeML DSL to SQL"
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Path to DSL file. Reads from stdin if omitted.",
    )
    args = parser.parse_args(argv)

    if args.source:
        with open(args.source, "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()

    model = parse(text)
    sql = compile_sql(model)
    sys.stdout.write(sql)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

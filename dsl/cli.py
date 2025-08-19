import argparse
import sys

from lark.exceptions import LarkError

from .parser import ComputeKernel, TrainModel, compile_sql, parse


def main(argv: list[str] | None = None) -> int:
    """Compile DeclarativeML DSL to SQL."""
    parser = argparse.ArgumentParser(
        description="Compile DeclarativeML DSL to SQL",
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Path to DSL file. Reads from stdin if omitted.",
    )
    args = parser.parse_args(argv)

    if args.source:
        try:
            with open(args.source, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            print(
                f"Failed to read source file '{args.source}': {exc}",
                file=sys.stderr,
            )
            return 1
    else:
        text = sys.stdin.read()

    try:
        model: TrainModel | ComputeKernel = parse(text)
        sql = compile_sql(model)
    except (LarkError, ValueError) as exc:
        print(f"Failed to compile DSL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - safety net
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    # Print the generated SQL with a trailing newline to ensure a clean output
    # when redirecting to files or piping to other commands.
    sys.stdout.write(sql + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

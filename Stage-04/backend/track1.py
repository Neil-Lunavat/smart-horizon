#!/usr/bin/env python3
"""Temporary Track 1 model: return the bundled sample output CSV."""

from pathlib import Path
import shutil
import sys


def main() -> None:
    # The uploaded path is intentionally accepted now so the interface is
    # ready for the real model, even though this placeholder ignores it.
    if len(sys.argv) != 2:
        raise SystemExit("Usage: track1.py <input.csv>")

    sample_output = Path(__file__).resolve().parent / "sample" / "sample_output.csv"
    if not sample_output.is_file():
        raise SystemExit(f"Sample output not found: {sample_output}")

    with sample_output.open("rb") as source:
        shutil.copyfileobj(source, sys.stdout.buffer)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

DEFAULT_LOG_DIR = Path("outputs/synthetic/logs")
PATTERN = re.compile(r"R\^2=([0-9.+-Ee]+)")
FILENAME_PATTERN = re.compile(r"q(?P<q>[\d.]+)_m(?P<m>\d+)")


def parse_log(path: Path):
    match = FILENAME_PATTERN.search(path.stem)
    if not match:
        return None
    q = float(match.group("q"))
    m = int(match.group("m"))
    r2 = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match_r2 = PATTERN.search(line)
        if match_r2:
            r2 = float(match_r2.group(1))
            break
    if r2 is None:
        return None
    return {"q": q, "m": m, "R2": r2}


def summarize_logs(log_dir: Path, output_csv: Path | None = None):
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(log_dir.glob("q*_m*.log")):
        parsed = parse_log(path)
        if parsed:
            rows.append(parsed)
    if not rows:
        raise SystemExit("No R^2 entries found in logs")
    out_path = output_csv or log_dir / "r2_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["q", "m", "R2"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return rows


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Extract logits-vs-Hamming R^2 values from synthetic logs."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory containing q*_m*.log files. Default: {DEFAULT_LOG_DIR}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <log-dir>/r2_summary.csv.",
    )
    args = parser.parse_args(argv)
    summarize_logs(args.log_dir, args.output_csv)


if __name__ == "__main__":
    main()



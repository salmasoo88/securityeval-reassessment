"""Restore pinned benchmark source text into the public feature snapshot."""
import argparse
import csv
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    expected = '6f4fb70f782c6d47b02ea24341e8ef8c1eb04a6a'
    actual = subprocess.check_output(
        ['git', '-C', str(args.benchmark), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if actual != expected:
        raise ValueError('Check out the benchmark commit documented in README.md')
    snapshot = Path(__file__).resolve().parents[1] / 'data' / 'feature_table.csv'
    with snapshot.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames + ['source_text']
        rows = list(reader)
    for row in rows:
        source = subprocess.check_output([
            'git', '-C', str(args.benchmark), 'show',
            expected + ':' + row['source_filename'],
        ])
        row['source_text'] = source.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')
    with args.out.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()


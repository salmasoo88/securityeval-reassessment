"""Recover the Linux CodeQL CSVs and compare both alert and file-level results."""
import argparse
import base64
from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import functools
import gzip
import io
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--serial", type=Path, required=True)
parser.add_argument("--benchmark", type=Path, required=True)
parser.add_argument("--query-root", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--git", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--generator", choices=("Copilot", "InCoder"))
args = parser.parse_args()

serial = args.serial.read_text(encoding="utf8", errors="replace")
assert "x86_64 GNU/Linux" in serial and "Python 3.9.4" in serial
assert re.search(r'"version"\s*:\s*"2\.10\.0"', serial)
if not args.generator:
    assert "CODEQL_STUDY_FINISHED" in serial
manifest = json.loads(args.manifest.read_text(encoding="utf8"))
archival_revision = "ba11cdd3e04035a9057412d7e677b1d4f5e2e41a"
current_revision = manifest["source_commit"]
git = [str(args.git), "-C", str(args.benchmark)]
args.out.mkdir(parents=True, exist_ok=True)
output = dict(archival_commit=archival_revision, current_commit=current_revision,
              query_commit=manifest["query_commit"],
              database_extraction="Windows" if "PORTABLE_WINDOWS_EXTRACTED_DATABASES" in serial else "Linux",
              query_evaluation="Linux", generators={})
differences = []

def query_cwe(name):
    return "CWE-" + re.search(r"(\d{3})", name)[1]

def read(path):
    with path.open(encoding="utf8", newline="") as f:
        return list(csv.reader(f))

def ident(path):
    return tuple(path.replace("\\", "/").split("/")[-2:])

def signature(row):
    return (row[0], row[2], *ident(row[4]), *map(int, row[5:9]))

@functools.lru_cache(maxsize=None)
def line_map(generator, cwe, filename):
    path = f"Testcases_{generator}/{cwe}/{filename}"
    before = subprocess.check_output(git + ["show", f"{archival_revision}:{path}"]).splitlines()
    after = subprocess.check_output(git + ["show", f"{current_revision}:{path}"]).splitlines()
    return {a + k + 1: b + k + 1 for a,b,count in
            SequenceMatcher(None, before, after, autojunk=False).get_matching_blocks()
            for k in range(count)}

for generator in ([args.generator] if args.generator else ("Copilot", "InCoder")):
    lower = generator.lower()
    database_marker = (f"DATABASE_REUSED_{generator}" if output["database_extraction"] == "Windows"
                       else f"DATABASE_EXIT_{generator}:0")
    assert database_marker in serial and f"ANALYZE_EXIT_{generator}:0" in serial
    match = re.search(rf"BEGIN_CODEQL_{generator}\s+([A-Za-z0-9+/=\s]+?)END_CODEQL_{generator}", serial)
    assert match
    data = gzip.decompress(base64.b64decode("".join(match[1].split()), validate=True))
    section = serial.split(f"ANALYZE_START_{generator}", 1)[1].split(f"ANALYZE_EXIT_{generator}:0", 1)[0]
    evaluated = re.findall(r"\[(\d+)/40 eval[^\]]*\] Evaluation done; writing results to (\S+)\.bqrs\.", section)
    assert sorted(int(number) for number, _ in evaluated) == list(range(1, 41))
    assert len({path for _, path in evaluated}) == 40
    assert f"EVALUATION_EXIT_{generator}:0" in section and f"ALERT_QUERIES_{generator}:39" in section
    table = re.search(rf"BEGIN_TABLE_BQRS_{generator}\s+([A-Za-z0-9+/=\s]+?)END_TABLE_BQRS_{generator}", section)
    assert table
    (args.out / f"{lower}_diagnostic.bqrs").write_bytes(base64.b64decode("".join(table[1].split()), validate=True))
    (args.out / f"{lower}_linux.csv").write_bytes(data)
    fresh_rows = list(csv.reader(io.StringIO(data.decode("utf8"), newline="")))
    archive_root = args.benchmark / "Result" / f"testcases_{lower}"
    archive_rows = [(query_cwe(f.name), r)
                    for f in sorted(archive_root.glob("results_cwe_*.csv")) for r in read(f)]
    summaries = args.benchmark / "Result" / f"testcases_{lower}.csv"
    with summaries.open(encoding="utf8", newline="") as f:
        labels = {(r["CWEID"], r["SampleID"]): int(r["CodeQL"])
                  for r in csv.DictReader(f) if r["SampleID"].strip()}
    assert len(labels) == 130
    names = defaultdict(set)
    dirs = manifest["generators"][lower]["directories"]
    count_queries = 0
    for directory in dirs:
        for q in (args.query_root / directory).rglob("*.ql"):
            name = re.search(r"@name[ \t]+([^\r\n]+)", q.read_text(encoding="utf8"))
            assert name, q
            names[name[1].strip()].add(query_cwe(directory))
            count_queries += 1
    assert all(len(names[row[0]]) == 1 for row in fresh_rows)
    assert all(ident(row[4]) in labels for _, row in archive_rows)
    assert all(ident(row[4]) in labels for row in fresh_rows)
    archived_flags = {ident(row[4]) for _, row in archive_rows}
    fresh_flags = {ident(row[4]) for row in fresh_rows}
    archived_matched = {ident(row[4]) for cwe, row in archive_rows if cwe == ident(row[4])[0]}
    fresh_matched = {ident(row[4]) for row in fresh_rows if ident(row[4])[0] in names[row[0]]}
    summary_flags = {key for key, value in labels.items() if value}
    assert archived_matched == summary_flags
    archived = Counter(signature(row) for _, row in archive_rows)
    fresh = Counter(signature(row) for row in fresh_rows)
    mapped = Counter()
    for item, count in archived.items():
        name, severity, cwe, file, start, start_col, end, end_col = item
        locations = line_map(generator, cwe, file)
        if start in locations and end in locations:
            mapped[(name, severity, cwe, file, locations[start], start_col, locations[end], end_col)] += count
        else:
            mapped[item] += count
    report = dict(query_directories=len(dirs), query_files=count_queries, alert_queries=39,
                  fresh_alerts=fresh.total(), archive_alerts=archived.total(),
                  exact_location_matches=(fresh & archived).total(),
                  mapped_location_matches=(fresh & mapped).total(),
                  archive_flagged_files=len(archived_flags), fresh_flagged_files=len(fresh_flags),
                  archive_prompt_cwe_flags=len(archived_matched), fresh_prompt_cwe_flags=len(fresh_matched),
                  any_flag_fresh_only=sorted(fresh_flags - archived_flags),
                  any_flag_archive_only=sorted(archived_flags - fresh_flags),
                  prompt_cwe_fresh_only=sorted(fresh_matched - archived_matched),
                  prompt_cwe_archive_only=sorted(archived_matched - fresh_matched))
    output["generators"][lower] = report
    for comparison, expected in [("exact", archived), ("source_line_mapped", mapped)]:
        for side, missing in [("fresh_only", fresh - expected), ("archive_only", expected - fresh)]:
            for key, count in missing.items():
                differences.append((lower, comparison, side, *key, count))
    print(lower, json.dumps(report), flush=True)

(args.out / "comparison.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")
with (args.out / "alert_differences.csv").open("w", encoding="utf8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["generator","comparison","side","query","severity","cwe","file",
                "start_line","start_col","end_line","end_col","count"])
    w.writerows(differences)



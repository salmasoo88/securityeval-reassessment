"""Decode Linux guest scan outputs and compare with the archived reports."""
import argparse
import base64
from collections import Counter
import gzip
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--serial", type=Path, required=True)
parser.add_argument("--archived", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--generator", choices=("Copilot", "InCoder"))
args = parser.parse_args()
serial = args.serial.read_text(encoding="utf8", errors="replace")
if args.generator is None:
    assert "STUDY_FINISHED" in serial
assert re.search(r"Linux \S+ \S+ .*x86_64", serial)
args.out.mkdir(parents=True, exist_ok=True)

def identity(path, generator):
    parts = path.replace("\\", "/").split("/")
    anchor = parts.index(f"Testcases_{generator}")
    return "/".join(parts[anchor:])

def signature(result, generator):
    start, end = result["start"], result["end"]
    return (identity(result["path"], generator), result["check_id"],
            start["line"], start["col"], start["offset"],
            end["line"], end["col"], end["offset"], result["extra"]["severity"],
            result["extra"]["message"])

report = {}
for generator in ([args.generator] if args.generator else ("Copilot", "InCoder")):
    assert f"SCAN_EXIT_{generator}:0" in serial
    encoded = re.search(rf"BEGIN_SCAN_{generator}\s+([A-Za-z0-9+/=\s]+?)END_SCAN_{generator}", serial)
    assert encoded, generator
    fresh_data = gzip.decompress(base64.b64decode("".join(encoded.group(1).split()), validate=True))
    fresh = json.loads(fresh_data)
    archive = json.loads((args.archived / f"semgrep_{generator.lower()}.json").read_bytes())
    assert fresh["version"] == archive["version"] == "1.184.0"
    (args.out / f"semgrep_{generator.lower()}_linux.json").write_bytes(fresh_data)
    fa = Counter(signature(r, generator) for r in fresh["results"])
    aa = Counter(signature(r, generator) for r in archive["results"])
    def paths(doc):
        return {identity(p, generator) for p in doc["paths"]["scanned"]}
    fp, ap = paths(fresh), paths(archive)
    ff, af = {r[0] for r in fa}, {r[0] for r in aa}
    row = dict(fresh_version=fresh["version"], archive_version=archive["version"],
               fresh_engine=fresh.get("engine_requested"), archive_engine=archive.get("engine_requested"),
               fresh_scanned=len(fp), archive_scanned=len(ap),
               fresh_scanned_only=sorted(fp-ap), archive_scanned_only=sorted(ap-fp),
               fresh_errors=len(fresh["errors"]), archive_errors=len(archive["errors"]),
               fresh_skipped_rule_entries=len(fresh.get("skipped_rules", [])),
               archive_skipped_rule_entries=len(archive.get("skipped_rules", [])),
               fresh_findings=fa.total(), archive_findings=aa.total(),
               exact_matched_findings=(fa & aa).total(),
               fresh_only_findings=[dict(signature=k, count=v) for k,v in sorted((fa-aa).items())],
               archive_only_findings=[dict(signature=k, count=v) for k,v in sorted((aa-fa).items())],
               fresh_flagged=len(ff), archive_flagged=len(af),
               fresh_flagged_only=sorted(ff-af), archive_flagged_only=sorted(af-ff),
               fresh_rule_ids_with_findings=len({r[1] for r in fa}),
               archive_rule_ids_with_findings=len({r[1] for r in aa}))
    report[generator.lower()] = row
    print(generator, {k:v for k,v in row.items() if not k.endswith("_only_findings")})
(args.out / "linux_scan_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")


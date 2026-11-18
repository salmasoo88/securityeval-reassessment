"""Run explicitly installed analyzers on the two pinned benchmark completion sets."""
import argparse
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

GENERATORS = ("Copilot", "InCoder")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("securityeval-root", "rules-root", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    for package, expected in (("bandit", "1.9.4"), ("semgrep", "1.184.0")):
        if version(package) != expected:
            raise ValueError(f"{package} {expected} required")
    rules = args.rules_root / "python"
    if not rules.is_dir():
        raise FileNotFoundError(rules)
    targets = [args.securityeval_root / f"Testcases_{g}" for g in GENERATORS]
    for target in targets:
        if not target.is_dir():
            raise FileNotFoundError(target)
    args.out.mkdir(parents=True, exist_ok=False)
    for g, target in zip(GENERATORS, targets):
        subprocess.run([sys.executable, "-m", "bandit", "--exit-zero", "-r", str(target),
                        "-f", "json", "-o", str(args.out / f"bandit_{g.lower()}.json")], check=True)
    for g, target in zip(GENERATORS, targets):
        subprocess.run(["semgrep", "scan", "--metrics=off", "--disable-version-check", "--jobs=1",
                        "--config", str(rules), "--json", "--output",
                        str(args.out / f"semgrep_{g.lower()}.json"), str(target)], check=True)


if __name__ == "__main__":
    main()


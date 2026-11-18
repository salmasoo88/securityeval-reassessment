# SecurityEval Reassessment

Reproducible Evaluation of AI-Generated Python Vulnerability Detection

## Process abstract

This project evaluates how scanner provenance and evaluation design affect vulnerability detection on AI-generated Python. It uses 260 legacy SecurityEval completions from Copilot and InCoder across 130 paired prompts. Scanner findings are attributed to individual files. Five classifiers are compared with static-analysis baselines using prompt-grouped and CWE-grouped evaluation, training-only preprocessing, and nested threshold selection. Feature ablations, cross-generator transfer, formatting sensitivity, paired statistical comparisons, and scanner reproduction checks examine effects on review workload and missed benchmark positives. Findings concern this legacy cohort. Independent weakness adjudication and validation on current generators remain necessary.

## Dataset and setup

The public data/feature_table.csv contains 260 rows and 70 columns of benchmark labels, identifiers, scanner flags, and numeric features. codeql_flagged represents the benchmark prompt-CWE definition. Source text is restored from pinned upstream Git objects with the supplied script.

Use Python 3.12 in a virtual environment. The reference evaluation used Python 3.12.14.

```sh
python -m pip install -r code/requirements.txt
python -m unittest discover -s code -v
git clone https://github.com/s2e-lab/SecurityEval.git SecurityEval
git -C SecurityEval checkout --detach 6f4fb70f782c6d47b02ea24341e8ef8c1eb04a6a
python code/restore_source_text.py --benchmark SecurityEval --out feature_table.csv
python code/evaluate.py --feature-table feature_table.csv --out runs/prompt
python code/evaluate.py --feature-table feature_table.csv --out runs/cwe --group-by cwe
python code/independent_validation.py --table feature_table.csv --benchmark SecurityEval --out runs/verified runs/prompt runs/cwe
```

Use fresh output paths. Evaluation saves predictions, scores, folds, thresholds, ablations, transfer results, and statistical summaries.

## Scanner checks and interpretation

Run scripts with --help for their required inputs. Scanner report comparisons require additional analyzer outputs or archived reports. Fresh Linux scans use Bandit 1.9.4, Semgrep 1.184.0, and semgrep/semgrep-rules at 16b9e6d45f10649f6a531ec4838ed9924372c3bc, using code/run_static_analysis.py. CodeQL comparison input requirements are listed by code/compare_codeql_linux.py --help.

Precision measures agreement with file-level benchmark labels. Confirming that alerts identify the intended weaknesses requires independent review. Classifier hyperparameters are fixed, while inner validation selects thresholds and, in the joint procedure, classifier identity.

Data derives from [SecurityEval](https://github.com/s2e-lab/SecurityEval) and scanner evidence. Consult the upstream repository for source provenance and applicable terms.

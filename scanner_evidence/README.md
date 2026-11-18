# Scanner evidence and experiment reproduction

## Supplied evidence

| Path | Purpose |
| --- | --- |
| `inputs/archived_scans/` | Four original Bandit/Semgrep JSON reports used to build the corrected model inputs |
| `inputs/pooled_scanner_inputs.csv` | Historical pooled scanner features used in the controlled attribution sensitivity |
| `scanner_evidence/bandit/` | Fresh Bandit reports retained from the reproduction |
| `scanner_evidence/semgrep/` | Fresh Linux reports, comparison summaries, serial captures, and scan provenance |
| `scanner_evidence/codeql/` | Fresh query CSVs, diagnostic CSVs, query manifest, Linux serial capture, comparison summary, and extraction-attempt log |

Archived CodeQL CSVs and manual summary labels are available in `SecurityEval/Result/` at the pinned benchmark revision. Use a full Git clone because CodeQL line mapping also reads an earlier revision. Query files come from the pinned upstream CodeQL repository. Download instructions follow below. Large third-party tool distributions and databases can be acquired or built separately.

Run all commands from the repository root, with the Python dependencies in the main README installed. The shell examples use a POSIX shell. On Windows, use the full path to `git.exe` for the CodeQL comparator's `--git` argument. Use fresh output paths.

## 1. Acquire inputs and rebuild the feature table

```sh
git clone https://github.com/s2e-lab/SecurityEval.git SecurityEval
git -C SecurityEval checkout --detach 6f4fb70f782c6d47b02ea24341e8ef8c1eb04a6a
python code/features/build_feature_table.py --securityeval-root SecurityEval --static-dir inputs/archived_scans --out feature_table.csv
```

The rebuild uses archived per-CWE CodeQL exports, including off-category alerts. That definition flags 66 files. The benchmark summary counts prompt-CWE matches and flags 44. `verify_results.py` below derives the summary comparator directly from the pinned summaries. Both definitions are baseline inputs, and CodeQL is excluded from learned features.

Alternatively, use `restore_source_text.py` from the main README to combine the public numeric snapshot with the pinned source. Both paths should reproduce the same 260 model-input records. The benchmark programs are parsed or scanned as data by these scripts.

## 2. Models, ablation, transfer and split sensitivity

```sh
python code/evaluate.py --feature-table feature_table.csv --out runs/prompt_42
python code/evaluate.py --feature-table feature_table.csv --out runs/cwe_42 --group-by cwe
python code/evaluate.py --feature-table feature_table.csv --out runs/prompt_101 --split-seed 101
python code/evaluate.py --feature-table feature_table.csv --out runs/prompt_202 --split-seed 202
python code/independent_validation.py --table feature_table.csv --benchmark SecurityEval --out runs/verified runs/prompt_42 runs/cwe_42 runs/prompt_101 runs/prompt_202
python code/verify_results.py --feature-table feature_table.csv --evaluation runs/prompt_42 --securityeval-root SecurityEval --out runs/prompt_42/codeql_sensitivity.csv
python code/verify_results.py --feature-table feature_table.csv --evaluation runs/cwe_42 --securityeval-root SecurityEval --out runs/cwe_42/codeql_sensitivity.csv
```

Each evaluation includes the fixed and nested classifiers, feature ablations, same-prompt and disjoint-prompt transfer, formatting sensitivity, paired bootstrap estimates, group-swap accuracy tests and Holm adjustment. The independent validator checks saved partitions, thresholds, decisions and arithmetic. The split seed changes partitions while estimator seeds remain fixed.

For matched training-size transfer, arrange the seed-42 run under `reference/prompt_cv` as required by the existing script.

```sh
mkdir -p reference/prompt_cv
cp feature_table.csv reference/feature_table.csv
cp runs/prompt_42/predictions.csv reference/prompt_cv/
cp runs/prompt_42/transfer_predictions.csv reference/prompt_cv/
python code/matched_transfer.py --code-root code --results-root reference --benchmark SecurityEval --archived-scans inputs/archived_scans --out runs/matched_transfer
python code/attribution_sensitivity.py --table feature_table.csv --pooled inputs/pooled_scanner_inputs.csv --prior runs --out runs/attribution
python code/review_workload.py --predictions runs/prompt_42/predictions.csv --benchmark SecurityEval --out runs/workload
mkdir -p runs/descriptives
python code/benchmark_descriptives.py --feature-table feature_table.csv --securityeval-root SecurityEval --out runs/descriptives
```

## 3. Replay the Linux Semgrep comparison

```sh
python code/compare_semgrep.py --serial scanner_evidence/semgrep/semgrep_verified_invocation.txt --archived inputs/archived_scans --out runs/semgrep_replay
```

This decodes the saved scanner outputs and compares finding multisets and file coverage. It does not execute Semgrep again. The expected matches are all 83 Copilot and 116 InCoder findings and all 130 scanned files per generator. See `semgrep/PROVENANCE.md` for the pinned image digest, rules revision, invocation, and equivalence criteria. `semgrep_verified_invocation.txt` is the complete verified scan capture used by this replay.

For a new Linux scan, install Bandit 1.9.4 and Semgrep 1.184.0 in a separate scanner environment, then use the pinned rules.

```sh
python -m pip install bandit==1.9.4 semgrep==1.184.0
git clone https://github.com/semgrep/semgrep-rules.git semgrep-rules
git -C semgrep-rules checkout --detach 16b9e6d45f10649f6a531ec4838ed9924372c3bc
python code/run_static_analysis.py --securityeval-root SecurityEval --rules-root semgrep-rules --out runs/fresh_scans
```

The verified historical Linux scan used `--x-rule-validation=none` to skip preliminary CLI rule validation. The helper above uses ordinary CLI validation. The exact historical command is in the provenance file if reproducing that invocation is necessary. Compare new findings on the stated file/rule/location/severity/message criteria, allowing local path prefixes and runtime metadata to differ.

## 4. Replay the complete CodeQL query-suite comparison

```sh
git clone https://github.com/github/codeql.git codeql-queries
git -C codeql-queries checkout --detach 56786790fcf88b79d1d26b946ff36dfe2d6f33ac
python code/compare_codeql_linux.py --serial scanner_evidence/codeql/guest_serial.txt --benchmark SecurityEval --query-root codeql-queries/python/ql/src/Security --manifest scanner_evidence/codeql/query_manifest.json --git "$(command -v git)" --out runs/codeql_replay
```

The comparator decodes the saved query outputs, checks that all 40 queries completed per generator, verifies the 39 alert exports, and reconstructs the diagnostic BQRS outputs from the log. It maps unchanged source lines between the archival export revision and the pinned benchmark snapshot. The archival 44 Copilot and 37 InCoder alerts all match after mapping. The fresh reports contain 45 and 40 alerts. Four added alerts change one InCoder any-alert flag and no prompt-CWE flags.

This is evidence replay. A new full scan additionally requires the official CodeQL CLI 2.10.0, Python 3.9.4, databases extracted from the pinned source, and the query checkout above. The query directories are listed per generator in `query_manifest.json`. The original query repository revision is unknown. The verified full-suite run used Linux query evaluation on Windows-extracted databases. The separate Linux extraction attempt stopped during pack resolution, so end-to-end Linux extraction/analysis equivalence remains unresolved. The retained extraction log records that limitation. Alert messages and data-flow paths are outside the historical comparison's matching criteria.

## Integrity and scope

`SHA256SUMS.json` records hashes of the added evidence and analysis scripts. Evidence JSON/CSV and logs retain their recorded findings and local scanner paths. The public data uses file-level benchmark labels. Independent alert adjudication and a modern-generator replication remain future work.



Linux Semgrep reproduction

Semgrep distribution: official semgrep/semgrep 1.184.0 Linux amd64 image.
Image manifest: sha256:83b9e9ee3df8ab1482dd7680cc78f25f6d9619a0caab7a687cf9eed31e88e4a1
SecurityEval commit: 6f4fb70f782c6d47b02ea24341e8ef8c1eb04a6a
Semgrep rules commit: 16b9e6d45f10649f6a531ec4838ed9924372c3bc
Guest: Alpine Linux x86_64 kernel 6.18.60-0-virt in QEMU 11.1.3.
Input bytes were read from Git objects and checked against the pinned blob IDs.
The Linux VM had no network adapter. The benchmark source was scanned as data.

The scan invocation used the Semgrep Python CLI shipped in the Linux image:

semgrep scan --x-rule-validation=none --metrics=off --disable-version-check --jobs=1 \
  --config /study/semgrep-rules/python --json --output RESULT.json \
  /study/SecurityEval/Testcases_GENERATOR

The absolute paths are guest-local. Run the command for Copilot and InCoder
from a directory containing sibling SecurityEval and semgrep-rules checkouts.
The experimental flag disables preliminary CLI rule validation to reduce
emulation overhead. The scan engine still parses the pinned rules.
The report comparison removes only the file path prefix before the generator
directory. It compares scanned and flagged file sets and the multiset of
findings by file, full rule ID, start/end line, column and byte offsets,
severity and message. Timing, platform paths and other run-specific metadata
are outside this equivalence check.

The two *_linux.json files are fresh scanner outputs. The comparison JSON
records exact matches and differences. The serial capture includes scan exit
statuses and compressed, base64-encoded copies of those outputs.
The feature-table verification JSON records the shared SHA-256 of the original
corrected feature table and the table rebuilt with these fresh Semgrep reports
and the archived Bandit reports. No model was refit for this check.


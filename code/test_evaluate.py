import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from evaluate import choose_threshold, confusion, folds, group_swap_p, linked_groups, measures, transform
from static_features import _norm_key, compute_static_features, validate_coverage, zero_row


class EvaluationTests(unittest.TestCase):
    def test_static_identity_does_not_pool_generators(self):
        with TemporaryDirectory() as temp:
            report = Path(temp) / "bandit.json"
            report.write_text(json.dumps({"errors": [], "results": [{
                "filename": r"C:\input\Testcases_Copilot\CWE-020\a.py",
                "issue_severity": "HIGH", "issue_confidence": "HIGH", "test_id": "B001",
            }]}), encoding="utf8")
            rows = compute_static_features([report], [])
            self.assertEqual(rows["copilot", "CWE-020", "a.py"]["bandit_n"], 1)
            self.assertNotIn(("incoder", "CWE-020", "a.py"), rows)
            self.assertEqual(_norm_key("/input/Testcases_InCoder/CWE-020/a.py"),
                             ("incoder", "CWE-020", "a.py"))
            self.assertEqual(zero_row()["sa_agreement"], 1)
            self.assertEqual(zero_row()["sa_union"], 0)

    def test_analyzer_errors_are_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            path.write_text('{"errors": ["parse failed"], "results": []}', encoding="utf8")
            with self.assertRaises(ValueError):
                compute_static_features([path], [])

    def test_missing_and_duplicate_scan_coverage_are_rejected(self):
        scanned = "Testcases_Copilot/CWE-020/a.py"
        expected = {("copilot", "CWE-020", "a.py")}
        with TemporaryDirectory() as temp:
            path = Path(temp) / "semgrep.json"
            for files, message in (([], "coverage mismatch"), ([scanned, scanned], "Duplicate")):
                path.write_text(json.dumps({"errors": [], "results": [], "paths": {"scanned": files}}))
                with self.assertRaisesRegex(ValueError, message):
                    validate_coverage([], [path], set() if files else expected)

    def test_transitive_duplicate_groups(self):
        df = pd.DataFrame({"prompt_group_key": ["a", "b", "b", "c", "d", "e"],
                           "source_text": ["x", " x ", "z", "z", "", ""]})
        groups = linked_groups(df)
        self.assertEqual(len(set(groups[:4])), 1)
        self.assertEqual(len(set(groups)), 3)

    def test_outer_and_inner_split_isolation(self):
        groups = np.repeat(np.arange(40), 2)
        y = np.tile([0, 1], 40)
        for train, test in folds(y, groups):
            for inner_train, valid in folds(y[train], groups[train]):
                self.assertFalse(set(groups[train[inner_train]]) & set(groups[train[valid]]))
                self.assertFalse(set(groups[train[valid]]) & set(groups[test]))

    def test_threshold_and_confusion(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.5, 0.6, 0.9])
        mcc, cutoff = choose_threshold(y, score, 0.5)
        pred = (score >= cutoff).astype(int)
        self.assertEqual(mcc, matthews_corrcoef(y, pred))
        self.assertEqual(list(confusion(y, pred)), [2, 0, 0, 2])
        self.assertEqual(choose_threshold(y, np.full(4, 0.5), 0.5), (0.0, 0.5))
        with self.assertRaises(ValueError):
            choose_threshold(y, np.full(4, np.nan), 0.5)

    def test_cluster_swap(self):
        self.assertEqual(group_swap_p([2, 2, 2]), 0.25)
        self.assertEqual(group_swap_p([2, -2]), 1.0)
        self.assertEqual(group_swap_p([0, 0]), 1.0)
        self.assertEqual(measures([2, 0, 0, 2])["mcc"], 1.0)

    def test_formatting_keeps_identifiers_docstrings_and_types(self):
        source = 'def f(x):\n    """doc"""\n    a = x  # type: int\n    return a\n'
        formatted = transform(source)
        self.assertIn('"""doc"""', formatted)
        self.assertIn("# type: int", formatted)
        self.assertIn("return a", formatted)
        self.assertIsNone(transform("return 0\n"))


if __name__ == "__main__":
    unittest.main()


"""
Appendix A.2 (21 structural/lexical metrics) + A.3 (25 risk-indicative
token counts) = 46 features total, computed directly from source text
and its Python AST.

Per 3.3.2: "Samples that fail to parse are handled gracefully, with
structural features set to zero and a parse-failure indicator retained
as an explicit signal" -- parse_ok is 0 and every other structural
feature is 0 in that case; the risk-token counts (which are plain
string searches, not AST-based) are still computed since they don't
require a successful parse.
"""
from __future__ import annotations

import ast
import math
import re
from collections import Counter
from pathlib import Path

RISK_TOKENS = [
    "eval(", "exec(", "pickle.", "os.system", "subprocess", "shell=True",
    "md5", "sha1", "random.", "yaml.load", "verify=False", "input(",
    "__import__", "marshal", "tempfile.mktemp", "assert", "chmod", "0777",
    "pyc", "cgi", "urlopen", "request.args", "format(", "% (", "os.popen",
]
assert len(RISK_TOKENS) == 25, len(RISK_TOKENS)

STRUCTURAL_NAMES = [
    "loc", "n_chars", "avg_line_len", "max_line_len", "comment_ratio",
    "blank_ratio", "parse_ok", "n_ast_nodes", "n_funcs", "n_calls",
    "n_imports", "n_if", "n_loops", "n_try", "n_attr", "n_strings",
    "n_binop", "cyclomatic", "max_depth", "ast_entropy", "keyword_risk_score",
]
ALL_FEATURE_NAMES = STRUCTURAL_NAMES + [f"tok_{i}" for i in range(len(RISK_TOKENS))]


def _risk_token_counts(source: str) -> list[int]:
    counts = []
    for tok in RISK_TOKENS:
        # literal substring count; tokens like "0777" and "% (" are not
        # valid Python identifiers so a regex-token approach would miss
        # them -- the appendix specifies "occurrences", so plain
        # substring counting is the most literal reading of A.3.
        counts.append(source.count(tok))
    return counts


def _line_metrics(source: str) -> dict:
    lines = source.splitlines()
    loc = len(lines)
    n_chars = len(source)
    line_lens = [len(l) for l in lines] or [0]
    avg_line_len = sum(line_lens) / loc if loc else 0.0
    max_line_len = max(line_lens) if line_lens else 0
    n_comment = sum(1 for l in lines if l.strip().startswith("#"))
    n_blank = sum(1 for l in lines if not l.strip())
    comment_ratio = n_comment / loc if loc else 0.0
    blank_ratio = n_blank / loc if loc else 0.0
    return {
        "loc": loc, "n_chars": n_chars, "avg_line_len": avg_line_len,
        "max_line_len": max_line_len, "comment_ratio": comment_ratio,
        "blank_ratio": blank_ratio,
    }


def _max_depth(node: ast.AST, depth: int = 0) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(_max_depth(c, depth + 1) for c in children)


def _ast_entropy(tree: ast.AST) -> float:
    kinds = [type(n).__name__ for n in ast.walk(tree)]
    total = len(kinds)
    if total == 0:
        return 0.0
    counts = Counter(kinds)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def _ast_metrics(source: str) -> dict:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return {"parse_ok": 0, **{k: 0 for k in STRUCTURAL_NAMES if k not in ("parse_ok",) and k not in _line_metrics("").keys()}}

    nodes = list(ast.walk(tree))
    n_funcs = sum(1 for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    n_calls = sum(1 for n in nodes if isinstance(n, ast.Call))
    n_imports = sum(1 for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom)))
    n_if = sum(1 for n in nodes if isinstance(n, ast.If))
    n_loops = sum(1 for n in nodes if isinstance(n, (ast.For, ast.AsyncFor, ast.While)))
    n_try = sum(1 for n in nodes if isinstance(n, ast.Try))
    n_attr = sum(1 for n in nodes if isinstance(n, ast.Attribute))
    n_strings = sum(
        1 for n in nodes
        if isinstance(n, ast.Constant) and isinstance(getattr(n, "value", None), str)
    )
    n_binop = sum(1 for n in nodes if isinstance(n, ast.BinOp))
    n_bool_and_compare = sum(1 for n in nodes if isinstance(n, (ast.BoolOp, ast.Compare)))

    # McCabe cyclomatic complexity = 1 + count(conditionals, loops, boolean
    # operators) per Appendix A.2's definition.
    cyclomatic = 1 + n_if + n_loops + sum(
        len(n.values) - 1 for n in nodes if isinstance(n, ast.BoolOp)
    )

    return {
        "parse_ok": 1,
        "n_ast_nodes": len(nodes),
        "n_funcs": n_funcs,
        "n_calls": n_calls,
        "n_imports": n_imports,
        "n_if": n_if,
        "n_loops": n_loops,
        "n_try": n_try,
        "n_attr": n_attr,
        "n_strings": n_strings,
        "n_binop": n_binop,
        "cyclomatic": cyclomatic,
        "max_depth": _max_depth(tree),
        "ast_entropy": _ast_entropy(tree),
    }


def compute_structural_features(source: str) -> dict:
    row = {}
    row.update(_line_metrics(source))
    row.update(_ast_metrics(source))
    if row.get("parse_ok", 0) == 0:
        # zero every structural feature except the line-level ones (those
        # are computable without a parse and the spec only requires
        # *structural* features to be zeroed on parse failure) and parse_ok.
        for name in STRUCTURAL_NAMES:
            if name not in row:
                row[name] = 0

    tok_counts = _risk_token_counts(source)
    row["keyword_risk_score"] = sum(tok_counts)
    for i, c in enumerate(tok_counts):
        row[f"tok_{i}"] = c

    # ensure every declared column is present and in order
    return {name: row.get(name, 0) for name in ALL_FEATURE_NAMES}


if __name__ == "__main__":
    # smoke test
    sample = Path(__file__).read_text()
    feats = compute_structural_features(sample)
    assert len(feats) == 46, len(feats)
    print(f"OK: {len(feats)} structural+token features computed")
    print({k: feats[k] for k in STRUCTURAL_NAMES[:6]})

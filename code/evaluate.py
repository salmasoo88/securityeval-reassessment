"""Grouped evaluation of SecurityEval completions. Never executes the samples."""
import argparse
import ast
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from model.pipeline import CLASSIFIER_NAMES, NUMERIC_FEATURES, make_pipeline
from structural_features import compute_structural_features

SEED = 42
KEY = ["generator", "cwe", "sample_id"]
FULL = ("static", "structural", "tokens")
FORMATTING = ("loc", "n_chars", "avg_line_len", "max_line_len", "comment_ratio", "blank_ratio")
ABLATIONS = {
    "static_only": ("static",),
    "structural_only": ("structural",),
    "tokens_only": ("tokens",),
    "static_plus_tokens": ("static", "tokens"),
    "static_plus_structural": ("static", "structural"),
    "structural_plus_tokens": ("structural", "tokens"),
    "minus_formatting": FULL,
}


def load_table(path):
    df = pd.read_csv(path, keep_default_na=False)
    identity = KEY + ["prompt_group_key"]
    if df.empty or df[identity].eq("").any().any() or df.duplicated(KEY).any():
        raise ValueError("Missing or duplicate sample identity")
    if not df["label"].isin([0, 1]).all() or df["label"].nunique() != 2:
        raise ValueError("Expected two binary label classes")
    if not (df["cwe"] + "::" + df["sample_id"]).equals(df["prompt_group_key"]):
        raise ValueError("Invalid prompt key")
    flags = ["bandit_flagged", "semgrep_flagged", "codeql_flagged"]
    if not df[flags].isin([0, 1]).all().all():
        raise ValueError("Expected binary analyzer flags")
    if not np.isfinite(df[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Invalid numeric features")
    return df


def linked_groups(df, base="prompt_group_key"):
    """Keep exact nonempty source duplicates and the base groups together."""
    parent = {g: g for g in df[base]}
    seen = {}

    def find(group):
        while parent[group] != group:
            group = parent[group]
        return group

    for group, source in zip(df[base], df.source_text):
        if source.strip():
            digest = hashlib.sha256(source.strip().encode()).hexdigest()
            earlier = seen.setdefault(digest, group)
            parent[find(group)] = find(earlier)
    return np.array([find(g) for g in df[base]])


def folds(y, groups, split_seed=SEED):
    if len(set(groups)) < 5:
        raise ValueError("Fewer than five groups")
    split = list(StratifiedGroupKFold(5, shuffle=True, random_state=split_seed)
                 .split(np.zeros(len(y)), y, groups))
    if sorted(np.concatenate([test for _, test in split])) != list(range(len(y))):
        raise ValueError("Expected one held-out prediction per row")
    for train, test in split:
        if set(groups[train]) & set(groups[test]) or len(set(y[train])) != 2 or len(set(y[test])) != 2:
            raise ValueError("Group overlap or a single-class split")
    return split


def confusion(y, pred):
    return np.bincount(2 * y + pred, minlength=4)


def measures(counts):
    tn, fp, fn, tp = map(int, counts)
    def ratio(n, d):
        return n / d if d else 0.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, accuracy=ratio(tn + tp, sum(counts)),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                specificity=ratio(tn, tn + fp), f1=ratio(2 * tp, 2 * tp + fp + fn),
                mcc=ratio(tp * tn - fp * fn, ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5))


def choose_threshold(y, scores, default):
    if not np.isfinite(scores).all():
        raise ValueError("Incomplete inner predictions")
    # Same search budget for margins and probabilities. Prefer the native cutoff on ties.
    candidates = np.unique(np.r_[default, np.quantile(scores, np.arange(0.05, 1, 0.05))])
    return max(((measures(confusion(y, (scores >= t).astype(int)))["mcc"], float(t))
                for t in candidates), key=lambda pair: (pair[0], -abs(pair[1] - default), pair[1]))


def fit(df, y, train, name, family=FULL, exclude=()):
    return make_pipeline(name, family, random_state=SEED, exclude_numeric=exclude).fit(df.iloc[train], y[train])


def scores(model, name, test):
    return model.decision_function(test) if name == "linear_svm" else model.predict_proba(test)[:, 1]


def transform(source):
    """Normalize formatting only on sources that compile and round-trip structurally."""
    try:
        before = ast.parse(source, type_comments=True)
        compile(before, "<sample>", "exec")
        after = ast.unparse(before) + "\n"
        if ast.dump(before) == ast.dump(ast.parse(after, type_comments=True)):
            return after
    except (SyntaxError, ValueError, RecursionError):
        pass
    return None


def group_swap_p(differences):
    """Exact two-sided cluster swap for accuracy under groupwise exchangeability."""
    distribution = {0: 1}
    for delta in map(int, differences):
        updated = {}
        for total, n in distribution.items():
            for shifted in (total - delta, total + delta):
                updated[shifted] = updated.get(shifted, 0) + n
        distribution = updated
    observed = abs(sum(differences))
    return sum(n for total, n in distribution.items() if abs(total) >= observed) / sum(distribution.values())


def compare(y, predictions, groups):
    indices = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    draws = np.random.default_rng(SEED).integers(len(indices), size=(5000, len(indices)))
    pairs = [("random_forest_nested", "union2"),
             ("random_forest_nested", "union3"),
             ("random_forest_fixed", "static_plus_tokens"),
             ("random_forest_fixed", "tokens_only"),
             ("random_forest_fixed", "minus_formatting")]
    pairs += [(name + "_nested", "random_forest_nested") for name in CLASSIFIER_NAMES if name != "random_forest"]
    rows = []
    for a, b in pairs:
        ca, cb = [np.array([confusion(y[idx], predictions[name][idx]) for idx in indices]) for name in (a, b)]
        diffs = ca[:, 0] + ca[:, 3] - cb[:, 0] - cb[:, 3]
        row = dict(a=a, b=b, p_cluster_accuracy=group_swap_p(diffs),
                   a_only_correct=int(((predictions[a] == y) & (predictions[b] != y)).sum()),
                   b_only_correct=int(((predictions[b] == y) & (predictions[a] != y)).sum()))
        sampled_a, sampled_b = ca[draws].sum(axis=1), cb[draws].sum(axis=1)
        point_a, point_b = measures(ca.sum(axis=0)), measures(cb.sum(axis=0))
        for metric in ("accuracy", "mcc", "recall", "precision", "f1"):
            delta = [measures(x)[metric] - measures(z)[metric] for x, z in zip(sampled_a, sampled_b)]
            row[f"delta_{metric}"] = point_a[metric] - point_b[metric]
            row[f"delta_{metric}_lo"], row[f"delta_{metric}_hi"] = np.quantile(delta, [0.025, 0.975])
        rows.append(row)
    adjusted = 0.0
    for rank, row in enumerate(sorted(rows, key=lambda row: row["p_cluster_accuracy"])):
        adjusted = max(adjusted, min(1.0, (len(rows) - rank) * row["p_cluster_accuracy"]))
        row["p_holm_accuracy"] = adjusted
    return pd.DataFrame(rows)


def run(df, out, group_by, split_seed=SEED):
    y = df.label.to_numpy()
    groups = linked_groups(df, group_by)
    outer = folds(y, groups, split_seed)
    oof = df[KEY + ["label", "prompt_group_key"]].copy()
    oof["group"] = groups
    oof["fold"] = -1
    oof["majority"] = -1
    for col in ("bandit", "semgrep", "codeql"):
        oof[col] = df[col + "_flagged"]
    oof["union2"] = oof[["bandit", "semgrep"]].max(axis=1)
    oof["union3"] = oof[["bandit", "semgrep", "codeql"]].max(axis=1)
    oof["intersection3"] = oof[["bandit", "semgrep", "codeql"]].min(axis=1)

    summaries, transfers, split_log, inner_predictions = [], [], [], []
    for fold, (train, test) in enumerate(outer):
        oof.loc[test, "fold"] = fold
        oof.loc[test, "majority"] = int(y[train].mean() >= 0.5)
        inner = folds(y[train], groups[train], split_seed)
        for inner_fold, (_, valid) in enumerate(inner):
            for idx in train[valid]:
                split_log.append({"row": int(idx), "outer_fold": fold, "inner_validation_fold": inner_fold})
        selection = []
        print(f"{group_by} outer fold {fold + 1}/5", flush=True)
        for name in CLASSIFIER_NAMES:
            native = 0.0 if name == "linear_svm" else 0.5
            val_scores = np.full(len(train), np.nan)
            for inner_train, valid in inner:
                model = fit(df, y, train[inner_train], name)
                val_scores[valid] = scores(model, name, df.iloc[train[valid]])
            inner_mcc, theta = choose_threshold(y[train], val_scores, native)
            inner_predictions.extend(dict(outer_fold=fold, classifier=name, row=int(row), label=int(y[row]), score=float(score))
                                     for row, score in zip(train, val_scores))
            model = fit(df, y, train, name)
            held = scores(model, name, df.iloc[test])
            oof.loc[test, name + "_score"] = held
            oof.loc[test, name + "_threshold"] = theta
            oof.loc[test, name + "_nested"] = (held >= theta).astype(int)
            oof.loc[test, name + "_fixed"] = (held >= native).astype(int)
            selection.append((inner_mcc, name, theta))
            if name == "random_forest":
                changed = []
                positions = []
                for idx in test:
                    source = transform(df.at[idx, "source_text"])
                    if source is not None:
                        row = df.loc[idx].to_dict()
                        row.update(compute_structural_features(source))
                        row["source_text"] = source
                        changed.append(row)
                        positions.append(idx)
                if changed:
                    oof.loc[positions, "format_score"] = scores(model, name, pd.DataFrame(changed))
        inner_mcc, selected, theta = max(selection, key=lambda row: row[0])
        oof.loc[test, "selected_name"] = selected
        oof.loc[test, "selected_nested"] = oof.loc[test, selected + "_nested"]
        summaries.extend(dict(fold=fold, classifier=name, inner_mcc=mcc, threshold=t, selected=name == selected)
                         for mcc, name, t in selection)
        for name, family in ABLATIONS.items():
            exclude = FORMATTING if name == "minus_formatting" else ()
            model = fit(df, y, train, "random_forest", family, exclude)
            oof.loc[test, name] = (scores(model, "random_forest", df.iloc[test]) >= 0.5).astype(int)
        for source, target in (("copilot", "incoder"), ("incoder", "copilot")):
            source_train = train[df.generator.iloc[train].to_numpy() == source]
            target_test = test[df.generator.iloc[test].to_numpy() == target]
            if set(groups[source_train]) & set(groups[target_test]) or len(set(y[source_train])) != 2:
                raise ValueError("Invalid transfer split")
            model = fit(df, y, source_train, "random_forest")
            held = scores(model, "random_forest", df.iloc[target_test])
            transfers.extend(dict(row=int(idx), fold=fold, source=source, target=target,
                                  n_train=len(source_train), label=y[idx], score=score)
                             for idx, score in zip(target_test, held))
    for source, target in (("copilot", "incoder"), ("incoder", "copilot")):
        train = np.flatnonzero(df.generator.to_numpy() == source)
        test = np.flatnonzero(df.generator.to_numpy() == target)
        model = fit(df, y, train, "random_forest")
        held = scores(model, "random_forest", df.iloc[test])
        transfers.extend(dict(row=int(idx), fold=-1, source=source, target=target,
                              n_train=len(train), label=y[idx], score=score)
                         for idx, score in zip(test, held))
    names = ["majority", "bandit", "semgrep", "codeql", "union2", "union3", "intersection3",
             "selected_nested", *ABLATIONS]
    names += [name + suffix for name in CLASSIFIER_NAMES for suffix in ("_fixed", "_nested")]
    predictions = {name: oof[name].to_numpy(dtype=int) for name in names}
    metrics = []
    for name, pred in predictions.items():
        row = dict(method=name, **measures(confusion(y, pred)))
        classifier = name.rsplit("_", 1)[0]
        if classifier in CLASSIFIER_NAMES:
            continuous = oof[classifier + "_score"]
            # Foldwise ranking avoids comparing differently scaled margins across folds.
            row["mean_fold_roc_auc"] = np.mean([roc_auc_score(y[test], continuous.iloc[test]) for _, test in outer])
            row["mean_fold_ap"] = np.mean([average_precision_score(y[test], continuous.iloc[test]) for _, test in outer])
        metrics.append(row)
    transfer = pd.DataFrame(transfers)
    transfer["protocol"] = np.where(transfer.fold == -1, "same_prompt", "group_disjoint")
    transfer["prediction"] = (transfer.score >= 0.5).astype(int)
    transfer_metrics = [
        dict(protocol=protocol, source=source, target=target, n_test=len(part),
             **measures(confusion(part.label.to_numpy(), part.prediction.to_numpy())))
        for (protocol, source, target), part in transfer.groupby(["protocol", "source", "target"])
    ]
    keep = oof.format_score.notna().to_numpy()
    original = predictions["random_forest_fixed"][keep]
    formatted = (oof.loc[keep, "format_score"].to_numpy() >= 0.5).astype(int)
    format_metrics = [
        dict(version=name, n_compared=int(keep.sum()), agreement=float(np.mean(original == formatted)),
             mean_absolute_score_shift=float(np.mean(np.abs(oof.loc[keep, "format_score"] -
                                                           oof.loc[keep, "random_forest_score"]))),
             **measures(confusion(y[keep], pred)))
        for name, pred in (("original", original), ("formatted_static_frozen", formatted))
    ]
    for name, data in {
        "predictions": oof, "metrics": pd.DataFrame(metrics), "inner_folds": pd.DataFrame(split_log),
        "inner_selection": pd.DataFrame(summaries), "inner_predictions": pd.DataFrame(inner_predictions), "paired_statistics": compare(y, predictions, groups),
        "transfer_predictions": transfer, "transfer_metrics": pd.DataFrame(transfer_metrics),
        "format_metrics": pd.DataFrame(format_metrics),
    }.items():
        data.to_csv(out / f"{name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group-by", choices=["prompt_group_key", "cwe"], default="prompt_group_key")
    parser.add_argument("--split-seed", type=int, default=SEED, help="Seed for grouped folds only")
    args = parser.parse_args()
    df = load_table(args.feature_table)
    args.out.mkdir(parents=True, exist_ok=False)
    run(df, args.out, args.group_by, args.split_seed)
    metadata = dict(seed=args.split_seed, estimator_seed=SEED, bootstrap_seed=SEED, group_by=args.group_by, python=platform.python_version(),
                    feature_sha256=hashlib.sha256(args.feature_table.read_bytes()).hexdigest(),
                    packages={p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")})
    (args.out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()



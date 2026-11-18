"""Independently check saved grouped evaluations and derive reporting tables."""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from evaluate import CLASSIFIER_NAMES, KEY


def metrics(counts):
    tn, fp, fn, tp = np.moveaxis(np.asarray(counts, dtype=float), -1, 0)
    def div(n, d):
        return np.divide(n, d, out=np.zeros_like(n), where=d != 0)
    return dict(accuracy=div(tp + tn, tp + fp + fn + tn),
                mcc=div(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
                recall=div(tp, tp + fn), precision=div(tp, tp + fp),
                f1=div(2 * tp, 2 * tp + fp + fn))


def check_run(path):
    def read(name):
        return pd.read_csv(path / (name + ".csv"), float_precision="round_trip")
    pred, inner, selection, fold_ids = map(read, ("predictions", "inner_predictions", "inner_selection", "inner_folds"))
    labels = pred.label.to_numpy()
    assert len(pred) == 260 and int(labels.sum()) == 184 and not pred.duplicated(KEY).any()
    assert pred.groupby("prompt_group_key").fold.nunique().eq(1).all()
    assert pred.groupby("group").fold.nunique().eq(1).all()
    assert pred.groupby("fold").label.nunique().eq(2).all()
    threshold_rows = []
    for fold, outer in pred.groupby("fold"):
        train = pred.index[pred.fold != fold]
        memberships = fold_ids[fold_ids.outer_fold == fold].set_index("row").sort_index()
        assert memberships.index.equals(train)
        assert memberships.join(pred.group).groupby("group").inner_validation_fold.nunique().eq(1).all()
        candidates = []
        for name in CLASSIFIER_NAMES:
            part = inner[(inner.outer_fold == fold) & (inner.classifier == name)].set_index("row").sort_index()
            assert part.index.equals(train) and part.label.equals(pred.loc[train, "label"])
            native = 0. if name == "linear_svm" else .5
            grid = np.unique(np.r_[native, np.quantile(part.score, np.arange(.05, 1, .05))])
            score, _, cutoff = max((matthews_corrcoef(part.label, part.score.ge(value)),
                                    -abs(value - native), value) for value in grid)
            record = selection[(selection.fold == fold) & (selection.classifier == name)].iloc[0]
            assert np.isclose(score, record.inner_mcc) and cutoff == record.threshold
            candidates.append((name, score))
            assert outer[name + "_threshold"].eq(cutoff).all()
            assert outer[name + "_nested"].eq(outer[name + "_score"].ge(cutoff)).all()
            assert outer[name + "_fixed"].eq(outer[name + "_score"].ge(native)).all()
            if name == "random_forest":
                threshold_rows.append(dict(run=path.name, fold=fold, threshold=cutoff,
                    inner_selected_mcc=score, inner_fixed_mcc=matthews_corrcoef(part.label, part.score.ge(native)),
                    outer_fixed_mcc=matthews_corrcoef(outer.label, outer[name + "_fixed"]),
                    outer_nested_mcc=matthews_corrcoef(outer.label, outer[name + "_nested"]),
                    outer_positive=int(outer.label.sum()), outer_n=len(outer),
                    fixed_tp=int((outer.label * outer[name + "_fixed"]).sum()),
                    nested_tp=int((outer.label * outer[name + "_nested"]).sum()),
                    fixed_fp=int(((1 - outer.label) * outer[name + "_fixed"]).sum()),
                    nested_fp=int(((1 - outer.label) * outer[name + "_nested"]).sum())))
        winner = max(candidates, key=lambda item: item[1])[0]
        assert outer.selected_name.eq(winner).all()
        assert outer.selected_nested.equals(outer[winner + "_nested"])
    table = read("metrics")
    for row in table.itertuples(index=False):
        counts = confusion_matrix(labels, pred[row.method], labels=[0, 1]).ravel()
        assert tuple(counts) == (row.tn, row.fp, row.fn, row.tp)
        for key, value in metrics(counts).items():
            assert np.isclose(value, getattr(row, key))
    transfer = read("transfer_predictions")
    transfer_table = read("transfer_metrics").set_index(["protocol", "source", "target"])
    assert transfer.prediction.eq(transfer.score.ge(.5)).all()
    for (protocol, source, target), rows in transfer.groupby(["protocol", "source", "target"]):
        assert set(rows.row) == set(pred.index[pred.generator == target]) and rows.row.is_unique
        counts = confusion_matrix(rows.label, rows.prediction, labels=[0, 1]).ravel()
        reported = transfer_table.loc[protocol, source, target]
        assert tuple(counts) == tuple(reported[["tn", "fp", "fn", "tp"]])
        assert reported.n_test == len(rows)
        assert all(np.isclose(reported[k], v) for k, v in metrics(counts).items())
        for fold, part in rows.groupby("fold"):
            train = pred[pred.generator.eq(source) & pred.fold.ne(fold)]
            assert part.n_train.eq(len(train)).all()
            assert part.label.to_numpy().tolist() == pred.loc[part.row, "label"].tolist()
            if protocol == "group_disjoint":
                assert pred.loc[part.row, "fold"].eq(fold).all()
                assert not set(train.group) & set(pred.loc[part.row, "group"])
    formatted = pred[pred.format_score.notna()]
    assert len(formatted) > 0
    native, changed = formatted.random_forest_fixed, formatted.format_score.ge(.5)
    for row in read("format_metrics").itertuples(index=False):
        flags = native if row.version == "original" else changed
        counts = confusion_matrix(formatted.label, flags, labels=[0, 1]).ravel()
        assert tuple(counts) == (row.tn, row.fp, row.fn, row.tp)
        assert len(formatted) == row.n_compared and np.isclose(native.eq(changed).mean(), row.agreement)
        assert np.isclose((formatted.format_score - formatted.random_forest_score).abs().mean(),
                          row.mean_absolute_score_shift)
    groups = [g.index.to_numpy() for _, g in pred.groupby("group", sort=True)]
    weights = np.array([np.bincount(r, minlength=len(groups)) for r in
                        np.random.default_rng(42).integers(len(groups), size=(5000, len(groups)))])
    comparisons = read("paired_statistics")
    for row in comparisons.itertuples(index=False):
        a, b = [np.array([confusion_matrix(labels[g], pred.loc[g, m], labels=[0, 1]).ravel()
                         for g in groups]) for m in (row.a, row.b)]
        delta = (a - b)[:, [0, 3]].sum(axis=1)
        polynomial = np.array([1], dtype=object)
        for d in map(abs, delta):
            if d:
                factors = np.zeros(2 * d + 1, dtype=object)
                factors[[0, -1]] = 1
                polynomial = np.convolve(polynomial, factors)
        support = np.arange(len(polynomial)) - len(polynomial) // 2
        p = sum(polynomial[abs(support) >= abs(delta.sum())]) / sum(polynomial)
        assert np.isclose(p, row.p_cluster_accuracy)
        assert row.a_only_correct == int(((pred[row.a] == labels) & (pred[row.b] != labels)).sum())
        assert row.b_only_correct == int(((pred[row.b] == labels) & (pred[row.a] != labels)).sum())
        for name, va in metrics(weights @ a).items():
            diff = va - metrics(weights @ b)[name]
            assert np.allclose(np.quantile(diff, [.025, .975]),
                               [getattr(row, f"delta_{name}_{side}") for side in ("lo", "hi")])
            assert np.isclose(metrics(a.sum(axis=0))[name] - metrics(b.sum(axis=0))[name],
                              getattr(row, f"delta_{name}"))
    ordered = comparisons.sort_values("p_cluster_accuracy")
    adjusted = np.minimum(1, np.maximum.accumulate(ordered.p_cluster_accuracy.to_numpy() *
                                                    np.arange(len(ordered), 0, -1)))
    assert np.allclose(adjusted, ordered.p_holm_accuracy)
    return table.assign(run=path.name), threshold_rows, pred


def cohort(table, benchmark):
    frame = pd.read_csv(table, keep_default_na=False)
    observed = {}
    for generator in ("copilot", "incoder"):
        with (benchmark / "Result" / f"testcases_{generator}.csv").open(newline="", encoding="utf8") as file:
            for row in csv.DictReader(file):
                if row["SampleID"].strip():
                    key = generator, row["CWEID"], row["SampleID"]
                    assert key not in observed
                    observed[key] = int(row["Manual"]), int(row["CodeQL"])
    assert len(observed) == len(frame) == 260
    assert frame.groupby("prompt_group_key").generator.nunique().eq(2).all()
    assert frame.prompt_group_key.nunique() == 130 and frame.cwe.nunique() == 75
    assert frame.groupby("source_text").prompt_group_key.nunique().max() == 1
    assert Counter(frame.label) == {1: 184, 0: 76}
    for row in frame.itertuples(index=False):
        assert observed[row.generator, row.cwe, row.sample_id][0] == row.label
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    frame = cohort(args.table, args.benchmark)
    args.out.mkdir(parents=True, exist_ok=True)
    checked = [check_run(run) for run in args.runs]
    for run, (_, _, predictions) in zip(args.runs, checked):
        assert frame[KEY + ["label"]].equals(predictions[KEY + ["label"]])
        print(f"{run.name}: partitions, inner selection, thresholds, transfer, confusion matrices, bootstrap, swap and Holm verified")
    pd.concat([table for table, _, _ in checked]).to_csv(args.out / "verified_metrics.csv", index=False)
    pd.DataFrame([row for _, rows, _ in checked for row in rows]).to_csv(args.out / "threshold_diagnostics.csv", index=False)


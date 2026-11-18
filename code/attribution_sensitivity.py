"""Change only scanner-feature attribution on the original prompt partitions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import KEY, choose_threshold, confusion, fit, folds, linked_groups, load_table, measures, scores
from static_features import STATIC_FEATURE_NAMES


def run(table, pooled, prior, out):
    corrected = load_table(table)
    scanner_columns = KEY + STATIC_FEATURE_NAMES
    legacy = pd.read_csv(pooled, usecols=scanner_columns, keep_default_na=False)
    pooled_features = corrected[KEY].merge(legacy, on=KEY, validate="one_to_one")
    assert len(pooled_features) == len(corrected) and pooled_features[KEY].equals(corrected[KEY])
    alternate = corrected.copy()
    alternate[STATIC_FEATURE_NAMES] = pooled_features[STATIC_FEATURE_NAMES]
    y = corrected.label.to_numpy()
    groups = linked_groups(corrected)
    changed = (corrected[STATIC_FEATURE_NAMES] != alternate[STATIC_FEATURE_NAMES]).any(axis=1)
    out.mkdir(parents=True, exist_ok=False)
    pooled_features.to_csv(out / "pooled_scanner_inputs.csv", index=False)
    rows, summaries = [], []
    for seed in (42, 101, 202):
        expected = pd.read_csv(prior / f"prompt_{seed}/predictions.csv", float_precision="round_trip")
        assert expected[KEY + ["label"]].equals(corrected[KEY + ["label"]])
        for name, data in (("pooled", alternate), ("completion", corrected)):
            predicted = corrected[KEY + ["label", "prompt_group_key"]].copy()
            for fold, (train, test) in enumerate(folds(y, groups, seed)):
                inner = np.full(len(train), np.nan)
                for a, b in folds(y[train], groups[train], seed):
                    model = fit(data, y, train[a], "random_forest")
                    inner[b] = scores(model, "random_forest", data.iloc[train[b]])
                _, cutoff = choose_threshold(y[train], inner, .5)
                model = fit(data, y, train, "random_forest")
                values = scores(model, "random_forest", data.iloc[test])
                predicted.loc[test, ["fold", "cutoff"]] = fold, cutoff
                predicted.loc[test, "score"] = values
            assert predicted.fold.eq(expected.fold).all()
            predicted["fixed"] = predicted.score.ge(.5).astype(int)
            predicted["nested"] = predicted.score.ge(predicted.cutoff).astype(int)
            if name == "completion":
                assert np.array_equal(predicted.score, expected.random_forest_score)
                assert np.array_equal(predicted.cutoff, expected.random_forest_threshold)
                assert predicted.nested.eq(expected.random_forest_nested).all()
            rows.append(predicted.assign(seed=seed, attribution=name))
            for rule in ("fixed", "nested"):
                summaries.append(dict(seed=seed, attribution=name, rule=rule,
                    n_feature_rows_changed=int(changed.sum()),
                    **measures(confusion(y, predicted[rule].to_numpy()))))
            print(f"{seed} {name}: verified", flush=True)
    pd.concat(rows).to_csv(out / "attribution_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "attribution_metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("table", "pooled", "prior", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.table, args.pooled, args.prior, args.out)

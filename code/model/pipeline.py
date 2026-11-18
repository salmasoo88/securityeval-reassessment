"""
The hybrid detection pipeline (3.3.3 / 3.3.4): a single sklearn Pipeline
that fuses the 16 static-analysis features, the 46 structural/lexical
features, and a TF-IDF lexical representation of the raw source, feeding
one of five classifiers.

Crucially, the TF-IDF vectoriser AND the numeric-feature scaler are both
pipeline steps, not precomputed globally -- so when this pipeline is
handed to cross_val_predict (see train_eval.py), both are refit on each
training fold only. This is the "one thing in your favour" the supervisor
noted in Point 3: that part of the design was already correct, and this
file preserves it rather than accidentally precomputing TF-IDF outside
the CV loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from static_features import STATIC_FEATURE_NAMES  # noqa: E402
from structural_features import ALL_FEATURE_NAMES as STRUCTURAL_FEATURE_NAMES  # noqa: E402

NUMERIC_FEATURES = STATIC_FEATURE_NAMES + STRUCTURAL_FEATURE_NAMES  # 16 + 46 = 62
TEXT_COLUMN = "source_text"

FEATURE_FAMILIES = {
    "static": STATIC_FEATURE_NAMES,
    "structural": STRUCTURAL_FEATURE_NAMES,
    "tokens": [TEXT_COLUMN],  # TF-IDF lexical channel
}


def _select_text(df):
    return df[TEXT_COLUMN].fillna("")


def make_feature_union(numeric_cols: list[str], include_text: bool = True) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", StandardScaler(), numeric_cols))
    ct = ColumnTransformer(transformers, remainder="drop")

    if not include_text:
        return ct

    # Appendix A.4: regex tokeniser preserving identifiers and individual
    # operator characters as distinct tokens; unigrams+bigrams; sublinear
    # TF; min_df=2; vocabulary capped at 6000; case preserved.
    text_vectorizer = TfidfVectorizer(
        preprocessor=None,
        tokenizer=None,
        token_pattern=r"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=6000,
        lowercase=False,
    )

    from sklearn.pipeline import FeatureUnion

    class _DFColumnTransformer:
        """Wraps a ColumnTransformer (numeric) and a TfidfVectorizer (text)
        so both can live in one FeatureUnion operating on the same
        DataFrame input."""

    numeric_branch = Pipeline([
        ("select_numeric", FunctionTransformer(lambda df: df[numeric_cols], validate=False)),
        ("scale", StandardScaler()),
    ]) if numeric_cols else None

    text_branch = Pipeline([
        ("select_text", FunctionTransformer(_select_text, validate=False)),
        ("tfidf", text_vectorizer),
    ])

    branches = [("tokens", text_branch)]
    if numeric_branch is not None:
        branches.insert(0, ("numeric", numeric_branch))

    return FeatureUnion(branches)


def make_classifier(name: str, random_state: int = 42):
    if name == "logreg":
        return LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=random_state
        )
    if name == "linear_svm":
        return LinearSVC(class_weight="balanced", max_iter=10000, random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=random_state, n_jobs=1
        )
    if name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=random_state, n_jobs=1,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"unknown classifier {name!r}")


def make_pipeline(classifier_name: str, feature_families: tuple[str, ...] = ("static", "structural", "tokens"), random_state: int = 42, exclude_numeric: tuple[str, ...] = ()) -> Pipeline:
    numeric_cols: list[str] = []
    if "static" in feature_families:
        numeric_cols += STATIC_FEATURE_NAMES
    if "structural" in feature_families:
        numeric_cols += STRUCTURAL_FEATURE_NAMES
    include_text = "tokens" in feature_families

    features = make_feature_union([c for c in numeric_cols if c not in exclude_numeric], include_text=include_text)
    clf = make_classifier(classifier_name, random_state=random_state)
    return Pipeline([("features", features), ("clf", clf)])


CLASSIFIER_NAMES = ["logreg", "linear_svm", "random_forest", "xgboost", "gradient_boosting"]

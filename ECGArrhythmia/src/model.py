"""Training pipeline for AAMI 5-class ECG beat classification on MIT-BIH.

Loads the feature table produced by ``features.py``, splits records into the
literature-standard de Chazal et al. (2004) inter-patient DS1 (train) / DS2
(test) partition, tunes Random Forest, RBF-kernel SVM, and Gradient Boosting
under two class-imbalance strategies (class-weighting vs. combined
under/oversampling), builds a soft-voting ensemble of the three, evaluates
everything on the held-out DS2 patients, and writes a comparison table plus
the best model to ``results/``.

Usage (run from the ECGArrhythmia/ project root, not from src/ — see train.py):
    python train.py
    python train.py --quick   # tiny grids/cv for a fast correctness check
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

METADATA_COLUMNS = ["record", "sample", "symbol", "label"]
CLASS_LABELS = ["N", "S", "V", "F", "Q"]  # clinically meaningful order for reporting
RANDOM_STATE = 42

# Literature-standard inter-patient split (de Chazal et al., 2004). Records
# 102, 104, 107, and 217 are excluded: they're dominated by paced rhythms,
# which is the standard reason this split leaves them out of the AAMI
# 5-class inter-patient evaluation.
DS1_RECORDS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
               201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_RECORDS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
               213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# RR intervals longer than this are almost certainly gaps where MIT-BIH
# switched to rhythm-only annotation (e.g. ventricular flutter episodes in
# records 207/232) rather than a true beat-to-beat interval, so they're
# treated as missing rather than fed to the model as-is.
MAX_PHYSIOLOGICAL_RR_S = 3.0

# Cap on how many times SMOTE may multiply a minority class's original size
# (see `_oversample_strategy`) — prevents over-synthesizing from classes that
# are both rare and concentrated in just one or two patients.
MAX_OVERSAMPLE_MULTIPLIER = 8

# Power used to dampen "balanced" sample weights (see
# `compute_dampened_sample_weight`) — 1.0 reproduces sklearn's standard
# inverse-frequency weighting; lower values soften the weight given to
# extremely rare, patient-concentrated classes.
SAMPLE_WEIGHT_DAMPING_POWER = 0.5

PARAM_GRIDS = {
    "rf": {"n_estimators": [150, 300], "max_depth": [None, 20]},
    "svm": {"C": [1, 10], "gamma": ["scale", 0.01]},
    "gb": {"n_estimators": [100, 200], "max_depth": [3, 5]},
}
QUICK_PARAM_GRIDS = {
    "rf": {"n_estimators": [100], "max_depth": [None]},
    "svm": {"C": [1], "gamma": ["scale"]},
    "gb": {"n_estimators": [50], "max_depth": [3]},
}


def load_features(csv_path: Path = DEFAULT_FEATURES_CSV) -> pd.DataFrame:
    """Load the beat-level feature table written by ``features.py``.

    Args:
        csv_path: Path to the features CSV.

    Returns:
        Raw feature DataFrame, one row per beat.
    """
    return pd.read_csv(csv_path)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean known data artifacts in the raw feature table.

    Drops the four paced-rhythm records excluded by the standard
    inter-patient split, caps implausibly long RR intervals (annotation
    gaps, not real beat-to-beat intervals) to NaN, and median-imputes any
    remaining NaNs (first/last beat of a record has no RR neighbor).

    Args:
        df: Raw feature DataFrame from ``load_features``.

    Returns:
        Cleaned copy of the DataFrame, same shape minus excluded records.
    """
    excluded = {102, 104, 107, 217}
    df = df[~df["record"].isin(excluded)].copy()

    for col in ["rr_prev", "rr_next"]:
        df.loc[df[col] > MAX_PHYSIOLOGICAL_RR_S, col] = np.nan
    df.loc[df["rr_prev"].isna() | df["rr_next"].isna(), "rr_ratio"] = np.nan

    feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    return df


def ds_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition the feature table into DS1 (train) and DS2 (test) by record.

    Args:
        df: Cleaned feature DataFrame.

    Returns:
        (ds1_df, ds2_df) — disjoint by record, per the literature-standard
        inter-patient split, guaranteeing no patient appears in both.
    """
    ds1 = df[df["record"].isin(DS1_RECORDS)].copy()
    ds2 = df[df["record"].isin(DS2_RECORDS)].copy()
    return ds1, ds2


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a feature DataFrame into model inputs, labels, and patient groups.

    Args:
        df: Cleaned feature DataFrame (output of ``clean_features``).

    Returns:
        (X, y, groups) where X holds only feature columns, y is the AAMI
        class label, and groups is the record (patient) ID per row.
    """
    feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    return df[feature_cols], df["label"], df["record"]


def _undersample_strategy(y) -> dict:
    """Compute a RandomUnderSampler target that caps the majority class.

    Caps the majority class (N) at 4x the size of the next-largest class
    (floor of 5000), rather than collapsing it to match the rarest class —
    full balancing would both discard most of the available N data and (in
    the downstream SMOTE step) require synthesizing an impractically large
    number of minority samples.

    Args:
        y: Array-like of labels for the current training fold.

    Returns:
        Dict mapping the majority class label to its capped target count.
    """
    counts = Counter(y)
    majority_label = max(counts, key=counts.get)
    other_counts = [c for label, c in counts.items() if label != majority_label]
    target = min(counts[majority_label], max(4 * max(other_counts), 5000))
    return {majority_label: target}


def _oversample_strategy(y) -> dict:
    """Compute a SMOTE target that brings minority classes toward the majority count.

    Runs after ``_undersample_strategy`` in the same pipeline, so ``y`` here
    already reflects the capped majority count. Each minority class is
    oversampled toward that count, but capped at ``MAX_OVERSAMPLE_MULTIPLIER``
    times its own original size. Classes that are both rare and clustered in
    very few patients (e.g. F, where ~90% of beats come from a single
    record) don't have enough real diversity for SMOTE's interpolation to
    synthesize genuinely new information — fully matching the majority count
    just floods training with near-duplicate variants of that one patient's
    signal, which is what caused this model's initial F-class precision
    collapse (it learned that patient's idiosyncrasies, not a generalizable
    "fusion beat" signature). Capping the multiplier still helps the
    well-represented classes (S, V) approach balance while limiting that
    failure mode for the concentrated ones.

    Args:
        y: Array-like of labels for the current (already-undersampled) fold.

    Returns:
        Dict mapping each non-majority class label to the target count.
    """
    counts = Counter(y)
    majority_count = max(counts.values())
    return {
        label: min(majority_count, count * MAX_OVERSAMPLE_MULTIPLIER)
        for label, count in counts.items()
        if count < majority_count
    }


def compute_dampened_sample_weight(y, power: float = SAMPLE_WEIGHT_DAMPING_POWER) -> np.ndarray:
    """Compute per-sample class-balancing weights, dampened for extreme imbalance.

    Standard inverse-frequency ("balanced") weighting assigns weight
    proportional to ``n_samples / (n_classes * class_count)``. At the
    imbalance ratios in this dataset (Q can outnumber N by 1000x or more in
    a given training fold) that formula produces extreme outlier weights for
    the rarest classes, which pushes the model to obsess over a handful of
    training examples — often from a single patient — at the expense of
    generalizing elsewhere. Raising the ratio to a fractional power (default
    0.5, i.e. a square root) keeps the same ordering and direction of effect
    but compresses the extremes.

    Args:
        y: Array-like of labels to compute weights for.
        power: Damping exponent; 1.0 reproduces standard balanced weighting.

    Returns:
        Array of per-sample weights, same length and order as ``y``.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_total = len(y)
    n_classes = len(classes)
    weight_by_class = {
        cls: (n_total / (n_classes * count)) ** power
        for cls, count in zip(classes, counts)
    }
    return np.array([weight_by_class[label] for label in y])


def build_pipeline(model, strategy: str, random_state: int = RANDOM_STATE) -> ImbPipeline:
    """Build a scaling (+ optional resampling) pipeline around a classifier.

    Args:
        model: An unfitted scikit-learn classifier instance.
        strategy: Either "classweight" (scaling only; imbalance handled via
            sample_weight at fit time) or "smote" (scaling, then dynamic
            combined under/oversampling, then the classifier).
        random_state: Seed for the resamplers.

    Returns:
        An imbalanced-learn Pipeline ending in the given model as step "model".
    """
    steps = [("scaler", StandardScaler())]
    if strategy == "smote":
        steps.append(("undersample", RandomUnderSampler(sampling_strategy=_undersample_strategy, random_state=random_state)))
        steps.append(("oversample", SMOTE(sampling_strategy=_oversample_strategy, random_state=random_state, k_neighbors=3)))
    elif strategy != "classweight":
        raise ValueError(f"Unknown strategy: {strategy!r}")
    steps.append(("model", model))
    return ImbPipeline(steps)


def get_base_model(model_name: str, random_state: int = RANDOM_STATE, probability: bool = False, **params):
    """Construct an unfitted classifier of the given type.

    Class imbalance is intentionally NOT baked in here via ``class_weight``
    — both imbalance strategies in this module use ``sample_weight`` (for
    "classweight") or upstream resampling (for "smote") instead, so that a
    single code path works uniformly across all three model types (Gradient
    Boosting has no ``class_weight`` parameter) and the soft-voting ensemble.

    Args:
        model_name: One of "rf", "svm", "gb".
        random_state: Seed for the classifier.
        probability: For "svm" only — whether to enable probability estimates
            (needed for ROC-AUC and soft voting, but slows down fitting via
            internal cross-validated calibration, so it's left off during
            hyperparameter search and only turned on for the final refit).
        **params: Hyperparameters to set on the classifier.

    Returns:
        An unfitted scikit-learn classifier instance.
    """
    if model_name == "rf":
        return RandomForestClassifier(random_state=random_state, n_jobs=1, **params)
    if model_name == "svm":
        return SVC(kernel="rbf", probability=probability, random_state=random_state, **params)
    if model_name == "gb":
        return GradientBoostingClassifier(random_state=random_state, **params)
    raise ValueError(f"Unknown model_name: {model_name!r}")


def tune_model(
    model_name: str,
    strategy: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    param_grids: dict,
    cv_splits: int = 3,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Run GridSearchCV for one model under one imbalance-handling strategy.

    Uses StratifiedGroupKFold for the inner cross-validation so that, in
    addition to being class-stratified (as requested), no patient's beats
    are ever split across a fold's train/validation portions — the same
    leakage guard used for the outer DS1/DS2 split, applied consistently to
    hyperparameter selection too.

    Args:
        model_name: One of "rf", "svm", "gb".
        strategy: "classweight" or "smote" (see ``build_pipeline``).
        X_train: Training feature matrix (DS1 only).
        y_train: Training labels.
        groups_train: Patient (record) ID per training row.
        param_grids: Dict of model_name -> {hyperparam: [values]}.
        cv_splits: Number of cross-validation folds.
        random_state: Seed for the model, resamplers, and CV splitter.
        n_jobs: Parallel jobs for GridSearchCV.

    Returns:
        The fitted GridSearchCV object (``.best_estimator_``/``.best_params_``
        available).
    """
    model = get_base_model(model_name, random_state)
    pipeline = build_pipeline(model, strategy, random_state)
    param_grid = {f"model__{k}": v for k, v in param_grids[model_name].items()}

    fit_params = {}
    if strategy == "classweight":
        fit_params["model__sample_weight"] = compute_dampened_sample_weight(y_train)

    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipeline, param_grid, scoring="f1_macro", cv=cv, n_jobs=n_jobs, refit=True)
    grid.fit(X_train, y_train, groups=groups_train, **fit_params)
    return grid


def fit_final_pipeline(model_name: str, strategy: str, best_params: dict, X_train, y_train, random_state: int = RANDOM_STATE) -> ImbPipeline:
    """Build and fit one model's final tuned pipeline on the full training set.

    Args:
        model_name: One of "rf", "svm", "gb".
        strategy: "classweight" or "smote".
        best_params: Hyperparameters selected by ``tune_model`` (unprefixed,
            e.g. {"n_estimators": 300}).
        X_train: Full DS1 training feature matrix.
        y_train: Full DS1 training labels.
        random_state: Seed for the classifier and resamplers.

    Returns:
        A fitted Pipeline, ready to call ``.predict``/``.predict_proba`` on
        raw (unscaled) feature rows.
    """
    probability = model_name == "svm"  # needed for ROC-AUC / soft voting downstream
    model = get_base_model(model_name, random_state, probability=probability, **best_params)
    pipeline = build_pipeline(model, strategy, random_state)

    fit_params = {}
    if strategy == "classweight":
        fit_params["model__sample_weight"] = compute_dampened_sample_weight(y_train)

    pipeline.fit(X_train, y_train, **fit_params)
    return pipeline


class SoftVotingEnsemble:
    """Soft-voting ensemble that averages predict_proba across pre-fitted pipelines.

    Used instead of scikit-learn's VotingClassifier because each base
    pipeline needs different fit-time handling (a per-step ``sample_weight``
    for the class-weighted strategy, or an internal resampler for the SMOTE
    strategy) baked in already — VotingClassifier re-fits all of its
    estimators uniformly via a single ``fit(X, y, sample_weight=...)`` call
    and can't forward step-scoped fit params into nested Pipelines.
    """

    def __init__(self, fitted_pipelines: dict):
        """Args:
            fitted_pipelines: Dict of name -> already-fitted Pipeline, each
                supporting ``predict_proba`` with the same class ordering.
        """
        self.fitted_pipelines = fitted_pipelines
        self.classes_ = next(iter(fitted_pipelines.values())).classes_

    def predict_proba(self, X) -> np.ndarray:
        """Average predicted class probabilities across all base pipelines."""
        probs = [p.predict_proba(X) for p in self.fitted_pipelines.values()]
        return np.mean(probs, axis=0)

    def predict(self, X) -> np.ndarray:
        """Predict the class with the highest averaged probability."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def evaluate_model(model, X_test, y_test, class_labels: list = CLASS_LABELS) -> dict:
    """Compute the full evaluation suite for a fitted model on held-out data.

    Args:
        model: A fitted classifier/pipeline/ensemble exposing ``predict`` and
            ``predict_proba``.
        X_test: Held-out (DS2) feature matrix.
        y_test: Held-out true labels.
        class_labels: Class order to use for all reported metrics.

    Returns:
        Dict with keys: confusion_matrix (list of lists), classification_report
        (dict, per-class + macro/weighted averages), roc_auc_per_class (dict),
        roc_auc_macro (float), macro_f1 (float).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    binarizer = LabelBinarizer()
    binarizer.fit(class_labels)
    y_test_bin = binarizer.transform(y_test)
    proba_class_order = list(model.classes_)
    proba_reordered = y_proba[:, [proba_class_order.index(c) for c in class_labels]]

    roc_auc_per_class = {}
    for i, label in enumerate(class_labels):
        roc_auc_per_class[label] = float(roc_auc_score(y_test_bin[:, i], proba_reordered[:, i]))

    return {
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=class_labels).tolist(),
        "classification_report": classification_report(y_test, y_pred, labels=class_labels, output_dict=True, zero_division=0),
        "roc_auc_per_class": roc_auc_per_class,
        "roc_auc_macro": float(np.mean(list(roc_auc_per_class.values()))),
        "macro_f1": float(f1_score(y_test, y_pred, labels=class_labels, average="macro")),
    }


def run_full_comparison(
    features_csv: Path = DEFAULT_FEATURES_CSV,
    results_dir: Path = RESULTS_DIR,
    param_grids: dict = PARAM_GRIDS,
    cv_splits: int = 3,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run the full tuning + evaluation pipeline across all models and strategies.

    For each of Random Forest, SVM, and Gradient Boosting, under both the
    "classweight" and "smote" imbalance strategies: tunes hyperparameters via
    GridSearchCV on DS1, refits the best configuration on all of DS1, and
    evaluates on DS2. Also builds and evaluates a soft-voting ensemble of the
    three tuned models for each strategy. Writes a comparison table, the
    per-model metrics, and the single best model (by macro F1, tie-broken by
    class-V recall) to ``results_dir``.

    Args:
        features_csv: Path to the features CSV from ``features.py``.
        results_dir: Directory to write comparison/metrics/model artifacts to.
        param_grids: Hyperparameter grids per model name.
        cv_splits: Number of inner cross-validation folds for tuning.
        random_state: Seed used throughout.
        n_jobs: Parallel jobs for GridSearchCV.

    Returns:
        Comparison DataFrame with one row per (model, strategy), including a
        "voting" pseudo-model row per strategy.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    df = clean_features(load_features(features_csv))
    ds1_df, ds2_df = ds_split(df)
    X_train, y_train, groups_train = get_feature_matrix(ds1_df)
    X_test, y_test, _ = get_feature_matrix(ds2_df)

    rows = []
    all_metrics = {}
    for strategy in ["classweight", "smote"]:
        fitted_by_model = {}
        for model_name in ["rf", "svm", "gb"]:
            print(f"Tuning {model_name} [{strategy}]...")
            grid = tune_model(model_name, strategy, X_train, y_train, groups_train, param_grids, cv_splits, random_state, n_jobs)
            print(f"  best params: {grid.best_params_}  (cv macro-F1={grid.best_score_:.4f})")

            best_params = {k.removeprefix("model__"): v for k, v in grid.best_params_.items()}
            final_pipeline = fit_final_pipeline(model_name, strategy, best_params, X_train, y_train, random_state)
            fitted_by_model[model_name] = final_pipeline

            metrics = evaluate_model(final_pipeline, X_test, y_test)
            all_metrics[f"{model_name}_{strategy}"] = metrics
            rows.append({
                "model": model_name, "strategy": strategy,
                "macro_f1": metrics["macro_f1"], "roc_auc_macro": metrics["roc_auc_macro"],
                "recall_V": metrics["classification_report"]["V"]["recall"],
                "precision_V": metrics["classification_report"]["V"]["precision"],
                "best_params": best_params,
            })

        print(f"Building voting ensemble [{strategy}]...")
        ensemble = SoftVotingEnsemble(fitted_by_model)
        metrics = evaluate_model(ensemble, X_test, y_test)
        all_metrics[f"voting_{strategy}"] = metrics
        rows.append({
            "model": "voting", "strategy": strategy,
            "macro_f1": metrics["macro_f1"], "roc_auc_macro": metrics["roc_auc_macro"],
            "recall_V": metrics["classification_report"]["V"]["recall"],
            "precision_V": metrics["classification_report"]["V"]["precision"],
            "best_params": None,
        })

        joblib.dump(fitted_by_model, results_dir / f"fitted_pipelines_{strategy}.joblib")

    comparison = pd.DataFrame(rows).sort_values(["macro_f1", "recall_V"], ascending=False).reset_index(drop=True)
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    with open(results_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    best_key = f"{comparison.iloc[0]['model']}_{comparison.iloc[0]['strategy']}"
    print(f"\nBest model: {best_key} (macro F1={comparison.iloc[0]['macro_f1']:.4f}, recall_V={comparison.iloc[0]['recall_V']:.4f})")

    if comparison.iloc[0]["model"] == "voting":
        best_obj = SoftVotingEnsemble(joblib.load(results_dir / f"fitted_pipelines_{comparison.iloc[0]['strategy']}.joblib"))
    else:
        best_obj = joblib.load(results_dir / f"fitted_pipelines_{comparison.iloc[0]['strategy']}.joblib")[comparison.iloc[0]["model"]]
    joblib.dump(best_obj, results_dir / "best_model.joblib")
    with open(results_dir / "best_model_metadata.json", "w") as f:
        json.dump({"model": comparison.iloc[0]["model"], "strategy": comparison.iloc[0]["strategy"],
                    "macro_f1": comparison.iloc[0]["macro_f1"], "recall_V": comparison.iloc[0]["recall_V"]}, f, indent=2)

    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV, help="Path to features.csv from features.py")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Directory to write comparison/metrics/model artifacts to")
    parser.add_argument("--cv-splits", type=int, default=3, help="Number of inner CV folds for GridSearchCV")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV")
    parser.add_argument("--quick", action="store_true", help="Use tiny hyperparameter grids and 2-fold CV for a fast correctness check")
    args = parser.parse_args()

    param_grids = QUICK_PARAM_GRIDS if args.quick else PARAM_GRIDS
    cv_splits = 2 if args.quick else args.cv_splits

    comparison = run_full_comparison(args.features_csv, args.results_dir, param_grids, cv_splits, RANDOM_STATE, args.n_jobs)
    print("\n" + comparison.drop(columns=["best_params"]).to_string(index=False))


if __name__ == "__main__":
    main()

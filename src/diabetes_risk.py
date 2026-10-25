"""Loading, pipelines and evaluation helpers for the Pima Indians diabetes analysis."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RENAME = {"preg": "Pregnancies", "plas": "Glucose", "pres": "BloodPressure", "skin": "SkinThickness",
          "insu": "Insulin", "mass": "BMI", "pedi": "DiabetesPedigree", "age": "Age"}
FEATURES = list(RENAME.values())
# A value of 0 is physiologically impossible for these measurements: it encodes "not recorded".
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the OpenML ARFF exactly as published (zeros kept)."""
    text = Path(path).read_text(encoding="latin-1")
    names = re.findall(r"^@attribute\s+'?([^'\s]+)'?", text, flags=re.I | re.M)
    body = re.split(r"^@data\s*$", text, flags=re.I | re.M)[1]
    df = pd.read_csv(io.StringIO(body.strip()), header=None, names=names, comment="%")
    df = df.rename(columns=RENAME)
    df["diabetes"] = (df.pop("class") == "tested_positive").astype(int)
    return df


def zeros_to_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[ZERO_MEANS_MISSING] = out[ZERO_MEANS_MISSING].replace(0, np.nan)
    return out


def make_model(estimator, add_indicator: bool = True) -> Pipeline:
    """Median imputation (+ missingness indicators) and scaling, fit inside each training fold."""
    return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
                     ("scale", StandardScaler()), ("model", estimator)])


def threshold_for_sensitivity(y, proba, target: float) -> float:
    y = np.asarray(y)
    pos = np.sort(np.asarray(proba)[y == 1])[::-1]
    return float(pos[int(np.ceil(target * len(pos))) - 1])


def screening_metrics(y, flagged) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    f = np.asarray(flagged).astype(bool)
    tp, fn, tn, fp = (f & y).sum(), (~f & y).sum(), (~f & ~y).sum(), (f & ~y).sum()
    div = lambda a, b: a / b if b else float("nan")
    return {"sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "PPV": div(tp, tp + fp), "NPV": div(tn, tn + fn)}


def repeated_cv(model_factory, X, y, n_repeats=20, n_splits=5, target_sensitivity=0.85, seed=0):
    """Out-of-fold probabilities per repeat, with flags at 0.5 and at a training-fold-chosen threshold."""
    y = np.asarray(y)
    results = []
    for r in range(n_repeats):
        proba = np.zeros(len(y))
        flag_target = np.zeros(len(y), dtype=bool)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for train, test in cv.split(X, y):
            Xtr, ytr = X.iloc[train], y[train]
            inner = cross_val_predict(model_factory(), Xtr, ytr, method="predict_proba",
                                      cv=StratifiedKFold(5, shuffle=True, random_state=seed + r))[:, 1]
            t = threshold_for_sensitivity(ytr, inner, target_sensitivity)
            model = model_factory().fit(Xtr, ytr)
            proba[test] = model.predict_proba(X.iloc[test])[:, 1]
            flag_target[test] = proba[test] >= t
        results.append({"proba": proba, "flag_05": proba >= 0.5, "flag_target": flag_target})
    return results


def bootstrap_ci(metric, y_true, y_score, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        stats.append(metric(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(metric(y_true, y_score)), float(lo), float(hi)

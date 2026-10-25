from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from diabetes_risk import (FEATURES, ZERO_MEANS_MISSING, load_raw, make_model, repeated_cv,
                           screening_metrics, sha256, threshold_for_sensitivity, zeros_to_missing)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "openml_37_diabetes.arff"
EXPECTED_SHA256 = "4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40"


def _toy(n=150, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(100, 20, n) for c in FEATURES})
    df.loc[:30, "Insulin"] = 0
    df.loc[:5, "Pregnancies"] = 0
    y = (df["Glucose"] + rng.normal(0, 15, n) > 110).astype(int).values
    return df, y


def test_zeros_to_missing_only_touches_physiological_columns():
    df, _ = _toy()
    out = zeros_to_missing(df)
    assert out["Insulin"].isna().sum() == 31
    assert out["Pregnancies"].isna().sum() == 0 and (out["Pregnancies"] == 0).sum() == 6
    assert set(ZERO_MEANS_MISSING) <= set(FEATURES)


def test_pipeline_imputes_inside_fit_with_indicators():
    df, y = _toy()
    X = zeros_to_missing(df)[FEATURES]
    model = make_model(LogisticRegression(max_iter=1000)).fit(X, y)
    # 8 features + 1 missingness indicator (Insulin)
    assert model.named_steps["scale"].n_features_in_ == 9
    assert np.isfinite(model.predict_proba(X)).all()


def test_threshold_and_metrics():
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.6])
    t = threshold_for_sensitivity(y, p, 0.75)
    assert t == 0.4
    assert screening_metrics(y, p >= t) == {"sensitivity": 0.75, "specificity": 0.5, "PPV": 0.75, "NPV": 0.5}


def test_repeated_cv_runs():
    df, y = _toy()
    res = repeated_cv(lambda: make_model(LogisticRegression(max_iter=1000)), zeros_to_missing(df)[FEATURES], y, n_repeats=2)
    assert len(res) == 2 and res[1]["proba"].shape == (len(y),)


@pytest.mark.skipif(not RAW.exists(), reason="OpenML file not present")
class TestRealData:
    def test_checksum(self):
        assert sha256(RAW) == EXPECTED_SHA256

    def test_structure_and_zero_counts(self):
        df = load_raw(RAW)
        assert df.shape == (768, 9) and df["diabetes"].sum() == 268
        zeros = {c: int((df[c] == 0).sum()) for c in ZERO_MEANS_MISSING}
        assert zeros == {"Glucose": 5, "BloodPressure": 35, "SkinThickness": 227, "Insulin": 374, "BMI": 11}

import argparse
import warnings
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEEDS         = [42, 2024, 12345, 99, 420]
HORIZONS      = [1, 3, 10, 25]
VAL_THRESHOLD = 3500   # train on <= 3500, validate on > 3500

# Base parameters shared across all horizons
BASE_PARAMS = {
    "objective"       : "regression",
    "metric"          : "rmse",
    "learning_rate"   : 0.015,
    "n_estimators"    : 4200,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq"    : 5,
    "lambda_l1"       : 0.1,
    "verbosity"       : -1,
    "n_jobs"          : -1,
}

# Horizon-specific overrides.
# Short horizons (1, 3) — high noise, strong regularisation needed:
#   fewer leaves, more samples per leaf, higher L2
# Long horizons (10, 25) — smoother signal, more model flexibility:
#   more leaves, fewer samples per leaf, lower L2
HORIZON_PARAMS = {
    1:  {"num_leaves": 70,  "min_child_samples": 250, "lambda_l2": 12.0},
    3:  {"num_leaves": 75,  "min_child_samples": 225, "lambda_l2": 11.0},
    10: {"num_leaves": 85,  "min_child_samples": 180, "lambda_l2":  9.0},
    25: {"num_leaves": 90,  "min_child_samples": 150, "lambda_l2":  8.0},
}

def get_params(horizon):
    """Merge base params with horizon-specific overrides."""
    return {**BASE_PARAMS, **HORIZON_PARAMS[horizon]}

NON_FEATURE_COLS = {
    "id", "code", "sub_code", "sub_category",
    "horizon", "ts_index", "weight", "y_target", "group_id",
}

# ---------------------------------------------------------------------------
# Kaggle metric
# ---------------------------------------------------------------------------

def kaggle_score(y_true, y_pred, weights):
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    weights = np.array(weights)
    num     = np.sum(weights * (y_true - y_pred) ** 2)
    den     = np.sum(weights * y_true ** 2)
    if den <= 0:
        return 0.0
    return float(np.sqrt(1.0 - np.clip(num / den, 0.0, 1.0)))


def build_features(df, horizon):
    df = df.copy()
    df = df.sort_values(["code", "sub_code", "sub_category",
                         "horizon", "ts_index"]).reset_index(drop=True)

    group_cols    = ["code", "sub_code", "sub_category", "horizon"]
    group_id_cols = ["code", "sub_code"]   # tighter group for lags

    # ── 1. feature_a derived features ────────────────────────────────────────
    # feature_a is a countdown (212 → 0) — time-to-expiry proxy.
    #
    # LEGAL normalisation: divide by 250 (a fixed constant > observed max of 213).
    # Why NOT transform("max"): scanning the full group for a max value
    # looks at future rows — a row at t=100 would "know" the value at t=3000.
    # Dividing by a fixed constant is always legal — no future data used.
    df["feature_a_pct"] = df["feature_a"] / 250.0

    # At time t, this is the largest countdown value seen so far (1..t)
    df["feature_a_exp_max"] = df.groupby(group_cols)["feature_a"].transform(
        lambda x: x.expanding().max()
    )
    # Fraction relative to expanding max — legal, uses only past data
    df["feature_a_exp_pct"] = df["feature_a"] / (df["feature_a_exp_max"] + 1e-7)

    # Rate of change of countdown — legal, just diff between consecutive rows
    df["feature_a_diff"] = df.groupby(group_cols)["feature_a"].transform(
        lambda x: x.diff()
    )

    # ── 2. Expanding means of y_target — CORRECT sequential stats ────────────
    # shift(1) before expanding() = only rows BEFORE current row inform it
    # This satisfies Y^t = pred(X[1:t]) strictly
    for col, grp in [("sub_category", "sub_category"),
                     ("code", "code"),
                     ("sub_code", "sub_code")]:
        df[f"{col}_exp_mean"] = (
            df.groupby(grp)["y_target"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        df[f"{col}_exp_std"] = (
            df.groupby(grp)["y_target"]
            .transform(lambda x: x.shift(1).expanding().std())
        )

    # ── 3. Spread / interaction features ────────────────────────────────────
    # These use original features — always legal
    df["d_al_am"]  = df["feature_al"] - df["feature_am"]
    df["r_al_am"]  = df["feature_al"] / (df["feature_am"].abs() + 1e-7)
    df["d_cg_by"]  = df["feature_cg"] - df["feature_by"]
    df["d_s_t"]    = df["feature_s"]  - df["feature_t"]
    df["d_al_cg"]  = df["feature_al"] - df["feature_cg"]

    # ── 4. Cross-sectional normalisation within ts_index ────────────────────
    # At time t, comparing across instruments is legal — same timestamp
    cs_cols = ["feature_al", "feature_am", "feature_cg",
               "feature_by", "d_al_am"]
    for col in cs_cols:
        grp = df.groupby("ts_index")[col]
        df[f"{col}_cs_mean"] = grp.transform("mean")
        df[f"{col}_cs_std"]  = grp.transform("std")
        df[f"{col}_z"]       = (
            (df[col] - df[f"{col}_cs_mean"]) /
            (df[f"{col}_cs_std"] + 1e-7)
        )
        df[f"{col}_rank"]    = grp.rank(pct=True)
        df[f"{col}_ts_min"]  = grp.transform("min")
        df[f"{col}_ts_max"]  = grp.transform("max")
        df[f"{col}_dist_min"]= df[col] - df[f"{col}_ts_min"]
        df[f"{col}_dist_max"]= df[f"{col}_ts_max"] - df[col]

    # ── 5. Lags of original features ─────────────────────────────────────────
    # shift(k) within group: row t sees value at t-k. Strictly legal.
    lag_feats = ["feature_al", "feature_am", "feature_cg",
                 "feature_by", "feature_s", "feature_t"]
    for feat in lag_feats:
        if feat not in df.columns:
            continue
        for lag in [1, 3, 5, 10]:
            df[f"{feat}_lag_{lag}"] = (
                df.groupby(group_cols)[feat].shift(lag)
            )
        # Difference (momentum)
        df[f"{feat}_diff_1"] = df.groupby(group_cols)[feat].diff(1)
        # Percent change
        df[f"{feat}_pct_1"]  = df.groupby(group_cols)[feat].pct_change(1)

    # ── 6. Rolling stats of original features ───────────────────────────────
    # Rolling on original features is legal — no y_target involved
    # min_periods=1 to avoid excessive NaN
    roll_feats = ["feature_al", "feature_am"]
    for feat in roll_feats:
        for w in [5, 10, 20]:
            df[f"{feat}_roll_mean_{w}"] = (
                df.groupby(group_cols)[feat]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            df[f"{feat}_roll_std_{w}"] = (
                df.groupby(group_cols)[feat]
                .transform(lambda x: x.rolling(w, min_periods=1).std())
            )

    # ── 7. EWM of original features ──────────────────────────────────────────
    spans = [3, 5] if horizon <= 3 else [5, 10]
    for feat in ["feature_al", "feature_am"]:
        for span in spans:
            df[f"{feat}_ewm_{span}"] = (
                df.groupby(group_cols)[feat]
                .transform(lambda x: x.ewm(span=span, adjust=False).mean())
            )

    # ── 8. Rolling volatility of cross-sectional z-scores ──────────────────
    for col in ["feature_al_z", "feature_cg_z"]:
        if col in df.columns:
            df[f"{col}_roll_std_10"] = (
                df.groupby(group_cols)[col]
                .transform(lambda x: x.rolling(10, min_periods=2).std())
            )
            df[f"{col}_roll_std_20"] = (
                df.groupby(group_cols)[col]
                .transform(lambda x: x.rolling(20, min_periods=2).std())
            )

    # ── 9. Time features — deterministic, always legal ───────────────────────
    df["ts_log"]     = np.log1p(df["ts_index"])
    df["ts_mod_30"]  = df["ts_index"] % 30
    df["ts_mod_90"]  = df["ts_index"] % 90
    df["ts_sin"]     = np.sin(2 * np.pi * df["ts_index"] / 365)
    df["ts_cos"]     = np.cos(2 * np.pi * df["ts_index"] / 365)
    df["ts_sin_100"] = np.sin(2 * np.pi * df["ts_index"] / 100)
    df["ts_cos_100"] = np.cos(2 * np.pi * df["ts_index"] / 100)
    df["ts_horizon"] = df["ts_index"] * df["horizon"]

    # ── 10. Sub-category dummies ──────────────────────────────────────────────
    sub_cat_dummies = pd.get_dummies(
        df["sub_category"], prefix="subcat", dtype=int
    )
    df = pd.concat([df, sub_cat_dummies], axis=1)

    df = df.fillna(0)
    return df


def train_horizon(train_path, test_path, horizon):
    print(f"\n{'='*60}")
    print(f"  HORIZON {horizon}")
    print(f"{'='*60}")

    # Load horizon slice
    train_raw = pd.read_parquet(train_path).query(
        f"horizon == {horizon}"
    ).copy()
    test_raw = pd.read_parquet(test_path).query(
        f"horizon == {horizon}"
    ).copy()

    # Add group_id and placeholder y_target/weight for test
    for df in [train_raw, test_raw]:
        df["group_id"] = (
            df["code"].astype(str) + "_" +
            df["sub_code"].astype(str) + "_" +
            df["sub_category"].astype(str) + "_" +
            df["horizon"].astype(str)
        )
    for col in ["y_target", "weight"]:
        if col not in test_raw.columns:
            test_raw[col] = 0.0

    # Step 1: Feature engineering on training data
    train_fe = build_features(train_raw.copy(), horizon)

    # Extract last expanding mean per group from training
    # These represent the complete historical mean up to end of training
    exp_cols = [c for c in train_fe.columns if "_exp_mean" in c or "_exp_std" in c]
    group_cols_key = ["code", "sub_code", "sub_category", "horizon"]

    last_exp_stats = (
        train_fe.sort_values(group_cols_key + ["ts_index"])
        .groupby(group_cols_key)[exp_cols]
        .last()
        .reset_index()
    )

    # Step 2: Feature engineering on test data
    # Expanding means computed on test data are meaningless (y_target=0)
    # so we will overwrite them with the correct last training values
    test_fe = build_features(
        pd.concat([train_raw, test_raw], ignore_index=True)
        .sort_values(group_cols_key + ["ts_index"])
        .reset_index(drop=True),
        horizon
    )
    test_fe = test_fe[test_fe["ts_index"] > train_raw["ts_index"].max()].copy()

    # Overwrite test expanding mean columns with last training values
    # This replaces the contaminated values with the correct historical means
    if exp_cols:
        test_fe = test_fe.drop(columns=exp_cols)
        test_fe = test_fe.merge(last_exp_stats, on=group_cols_key, how="left")
        # Fill any groups that appear only in test (no training history) with 0
        test_fe[exp_cols] = test_fe[exp_cols].fillna(0)

    print(f"  Expanding mean fix: {len(exp_cols)} columns corrected for test set")

    # Feature columns
    feat_cols = [c for c in train_fe.columns if c not in NON_FEATURE_COLS]
    h_params = get_params(horizon)
    print(f"  Features  : {len(feat_cols)}")
    print(f"  Config    : leaves={h_params['num_leaves']}  "
          f"min_child={h_params['min_child_samples']}  "
          f"L2={h_params['lambda_l2']}")

    # Time-based train/val split
    tr_mask  = train_fe["ts_index"] <= VAL_THRESHOLD
    val_mask = train_fe["ts_index"] >  VAL_THRESHOLD

    X_tr  = train_fe.loc[tr_mask,  feat_cols]
    y_tr  = train_fe.loc[tr_mask,  "y_target"]
    w_tr  = train_fe.loc[tr_mask,  "weight"]
    X_val = train_fe.loc[val_mask, feat_cols]
    y_val = train_fe.loc[val_mask, "y_target"]
    w_val = train_fe.loc[val_mask, "weight"]
    X_test = test_fe[feat_cols]
    ids    = test_fe["id"]

    print(f"  Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # Use raw weights directly.
    print(f"  Weight p50={w_tr.median():.1f}  p99={w_tr.quantile(0.99):.0f}  max={w_tr.max():.0f}")

    # Multi-seed ensemble
    val_preds  = np.zeros(len(X_val))
    test_preds = np.zeros(len(X_test))

    for i, seed in enumerate(SEEDS, 1):
        print(f"  Seed {i}/{len(SEEDS)} (seed={seed})...", end=" ", flush=True)

        model = lgb.LGBMRegressor(**{**get_params(horizon), "random_state": seed})
        model.fit(
            X_tr, y_tr,
            sample_weight      = w_tr.values,
            eval_set           = [(X_val, y_val)],
            eval_sample_weight = [w_val.values],
            callbacks          = [
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(period=99999),
            ],
        )

        val_preds  += model.predict(X_val)  / len(SEEDS)
        test_preds += model.predict(X_test) / len(SEEDS)
        print("done")

    h_score = kaggle_score(y_val, val_preds, w_val)
    print(f"\n  Horizon {horizon} local score: {h_score:.5f}")

    del train_raw, test_raw, train_fe, test_fe
    del X_tr, X_val
    gc.collect()

    return (
        pd.DataFrame({"id": ids.values, "prediction": test_preds}),
        list(y_val), list(val_preds), list(w_val),
        h_score,
    )


def run(train_path, test_path, output="predictions.csv"):

    all_test_preds = []
    all_y, all_p, all_w = [], [], []
    horizon_scores = {}

    for h in HORIZONS:
        test_df, y_val, p_val, w_val, h_score = train_horizon(
            train_path, test_path, h
        )
        all_test_preds.append(test_df)
        all_y.extend(y_val)
        all_p.extend(p_val)
        all_w.extend(w_val)
        horizon_scores[h] = h_score

    overall = kaggle_score(all_y, all_p, all_w)

    print(f"\n{'='*60}")
    print(f"  PER-HORIZON LOCAL SCORES (val: ts_index > {VAL_THRESHOLD})")
    for h, s in horizon_scores.items():
        print(f"    Horizon {h:2d}: {s:.5f}")
    print(f"  OVERALL LOCAL SCORE : {overall:.5f}")
    print(f"{'='*60}")

    submission = pd.concat(all_test_preds, axis=0, ignore_index=True)
    submission.to_csv(output, index=False)

    print(f"\nSubmission stats:")
    print(f"  Rows : {len(submission):,}")
    print(f"  mean : {submission['prediction'].mean():.6f}")
    print(f"  std  : {submission['prediction'].std():.6f}")
    print(f"  min  : {submission['prediction'].min():.4f}")
    print(f"  max  : {submission['prediction'].max():.4f}")
    print(f"  NaN  : {submission['prediction'].isna().sum()}")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.parquet")
    parser.add_argument("--test",  default="test.parquet")
    parser.add_argument("--out",   default="predictions.csv")
    args = parser.parse_args()
    run(args.train, args.test, args.out)
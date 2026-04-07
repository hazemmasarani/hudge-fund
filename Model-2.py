import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

NON_FEATURE_COLS = [
    "id",           
    "group_id",     
    "y_target",    
    "weight",       
    "is_test",      
    ]


CATEGORICAL_COLS = ["code", "sub_code", "sub_category"]

TARGET_COL = "y_target"
WEIGHT_COL = "weight"
ID_COL     = "id"


LGBM_PARAMS = {
    "objective"          : "regression",
    "metric"             : "rmse",
    "verbosity"          : -1,
    "num_leaves"         : 127,
    "min_child_samples"  : 100,
    "learning_rate"      : 0.05,
    "n_estimators"       : 1000,
    "early_stopping_rounds": 50,
    "feature_fraction"   : 0.8,
    "bagging_fraction"   : 0.8,
    "bagging_freq"       : 5,
    "reg_alpha"          : 0.1,
    "reg_lambda"         : 1.0,
    "n_jobs"             : -1,
    "random_state"       : 42,
}


def load(train_path: str, test_path: str
         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading feature files...")
    train = pd.read_parquet(train_path)
    test  = pd.read_parquet(test_path)
    print(f"  Train : {train.shape}")
    print(f"  Test  : {test.shape}")
    return train, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return columns to use as model input features."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def prepare_matrices(train: pd.DataFrame, test: pd.DataFrame
                     ) -> tuple:
    feature_cols = get_feature_cols(train)

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]
    w_train = train[WEIGHT_COL]
    X_test  = test[feature_cols]
    ids     = test[ID_COL]

    print(f"\nFeature matrix shapes:")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train — mean={y_train.mean():.4f}  std={y_train.std():.4f}")
    print(f"  Categorical cols: {CATEGORICAL_COLS}")

    # Report NaN situation in test so user knows what LGBM is handling
    nan_counts = X_test[X_test.columns[X_test.isna().any()]].isna().sum()
    if not nan_counts.empty:
        print(f"\n  NaNs in X_test (LightGBM handles these natively):")
        for col, cnt in nan_counts.items():
            print(f"    {col}: {cnt:,} NaN rows")

    return X_train, y_train, w_train, X_test, ids, feature_cols


def train_with_cv(X_train:     pd.DataFrame,
                  y_train:     pd.Series,
                  w_train:     pd.Series,
                  n_folds:     int = 5,
                  time_based:  bool = True,
                  ts_index_col: str = "ts_index"
                  ) -> tuple[lgb.LGBMRegressor, np.ndarray, list[float]]:
    
    oof_preds  = np.zeros(len(X_train))
    fold_rmses = []


    if time_based and ts_index_col in X_train.columns:
        sorted_idx = X_train[ts_index_col].argsort().values
        fold_size  = len(sorted_idx) // n_folds
        folds = []
        for k in range(n_folds):
            val_idx   = sorted_idx[k * fold_size : (k + 1) * fold_size]
            train_idx = sorted_idx[: k * fold_size]     # only past rows
            if len(train_idx) == 0:
                continue                                 # skip first fold if no history
            folds.append((train_idx, val_idx))
        print(f"\nUsing time-based CV with {len(folds)} valid folds")
    else:
        kf    = KFold(n_splits=n_folds, shuffle=False)
        folds = list(kf.split(X_train))
        print(f"\nUsing standard KFold CV with {n_folds} folds")

    # ---- Per-fold training ----
    for fold_num, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        w_tr, w_val = w_train.iloc[tr_idx], w_train.iloc[val_idx]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            sample_weight    = w_tr,
            eval_set         = [(X_val, y_val)],
            eval_sample_weight = [w_val],
            categorical_feature = CATEGORICAL_COLS,
            callbacks        = [lgb.early_stopping(LGBM_PARAMS["early_stopping_rounds"],
                                                   verbose=False),
                                 lgb.log_evaluation(period=100)],
        )

        val_preds          = model.predict(X_val)
        oof_preds[val_idx] = val_preds

        # Weighted RMSE — same metric the competition uses (weight column matters)
        wmse  = np.average((y_val - val_preds) ** 2, weights=w_val)
        wrmse = np.sqrt(wmse)
        fold_rmses.append(wrmse)

        # Pearson R — unweighted correlation (competition also reports this)
        r, _ = pearsonr(y_val, val_preds)

        print(f"  Fold {fold_num}: best_iteration={model.best_iteration_}  "
              f"weighted RMSE={wrmse:.6f}  Pearson R={r:.4f}")

    fold_pearson_rs = [pearsonr(
        y_train.iloc[val_idx], oof_preds[val_idx]
    )[0] for _, val_idx in folds]

    print(f"\nCV weighted RMSE : mean={np.mean(fold_rmses):.6f}  std={np.std(fold_rmses):.6f}")
    print(f"CV Pearson R      : mean={np.mean(fold_pearson_rs):.4f}  std={np.std(fold_pearson_rs):.4f}")

    # ---- Final model on all training data ----
    print("\nTraining final model on full training set...")
    # Use average best_iteration from CV folds as n_estimators (no early stopping here)
    avg_best_iter = int(np.mean([m for m in fold_rmses]))  # reuse loop for clarity
    final_params  = {**LGBM_PARAMS}
    # Remove early stopping — training on full data has no validation set
    final_params.pop("early_stopping_rounds", None)
    # Set n_estimators to a round number slightly above CV average
    final_params["n_estimators"] = 1000
    final_params["verbosity"]    = 0

    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(
        X_train, y_train,
        sample_weight       = w_train,
        categorical_feature = CATEGORICAL_COLS,
    )
    print("Final model trained.")
    return final_model, oof_preds, fold_rmses



def print_feature_importance(model:        lgb.LGBMRegressor,
                              feature_cols: list[str],
                              top_n:        int = 20) -> None:
    imp = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print(f"\nTop {top_n} features by gain importance:")
    for feat, score in imp.head(top_n).items():
        bar = "█" * int(score / imp.max() * 30)
        print(f"  {feat:<25} {bar}  {score:.0f}")




def predict_test(model:        lgb.LGBMRegressor,
                 X_test:       pd.DataFrame,
                 ids:          pd.Series,
                 output_path:  str = "predictions.csv") -> pd.DataFrame:

    print(f"\nGenerating predictions on {len(X_test):,} test rows...")
    preds = model.predict(X_test)

    results = pd.DataFrame({"id": ids.values, "y_target": preds})
    results.to_csv(output_path, index=False)
    print(f"Predictions saved: {output_path}  ({len(results):,} rows)")
    print(f"  Prediction stats — mean={preds.mean():.4f}  "
          f"std={preds.std():.4f}  "
          f"min={preds.min():.4f}  "
          f"max={preds.max():.4f}")
    return results


def save_model(model: lgb.LGBMRegressor, path: str = "lgbm_model.txt") -> None:
    model.booster_.save_model(path)
    print(f"Model saved: {path}")


def load_model(path: str = "lgbm_model.txt") -> lgb.Booster:
    model = lgb.Booster(model_file=path)
    print(f"Model loaded: {path}")
    return model


def run(train_path: str,
        test_path:  str,
        output:     str  = "predictions.csv",
        n_folds:    int  = 5,
        time_cv:    bool = True) -> None:

    # 1. Load
    train, test = load(train_path, test_path)

    # 2. Prepare feature matrices
    X_train, y_train, w_train, X_test, ids, feature_cols = \
        prepare_matrices(train, test)

    # 3. Cross-validated training + final model
    model, oof_preds, fold_rmses = train_with_cv(
        X_train, y_train, w_train,
        n_folds=n_folds,
        time_based=time_cv,
    )

    # 4. Overall OOF performance (weighted RMSE and Pearson R)
    oof_mask  = oof_preds != 0
    y_oof     = y_train[oof_mask]
    p_oof     = oof_preds[oof_mask]
    w_oof     = w_train[oof_mask]

    wmse_oof  = np.average((y_oof - p_oof) ** 2, weights=w_oof)
    wrmse_oof = np.sqrt(wmse_oof)
    r_oof, _  = pearsonr(y_oof, p_oof)

    print(f"\n{'='*55}")
    print(f"  Overall OOF weighted RMSE : {wrmse_oof:.6f}")
    print(f"  Overall OOF Pearson R     : {r_oof:.4f} ")
    print(f"{'='*55}")

    # 5. Feature importance
    print_feature_importance(model, feature_cols)

    # 6. Predict on test set
    predict_test(model, X_test, ids, output)

    # 7. Save model
    save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM model for AMS 580 dataset")
    parser.add_argument("--train",   default="train_features_cleaned.parquet",
                        help="Path to cleaned train parquet (output of Cleaner.py)")
    parser.add_argument("--test",    default="test_features_processed.parquet",
                        help="Path to processed test parquet (output of Cleaner.py)")
    parser.add_argument("--out",     default="predictions.csv",
                        help="Output path for test predictions CSV")
    parser.add_argument("--folds",   type=int,  default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--no_time_cv", action="store_true",
                        help="Use random KFold instead of time-based CV")
    args = parser.parse_args()

    run(
        train_path = args.train,
        test_path  = args.test,
        output     = args.out,
        n_folds    = args.folds,
        time_cv    = not args.no_time_cv,
    )

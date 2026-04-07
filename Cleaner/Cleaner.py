import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading data...")
    train = pd.read_parquet(train_path)
    test  = pd.read_parquet(test_path)
    print(f"  Train shape : {train.shape}")
    print(f"  Test  shape : {test.shape}")
    return train, test


def add_group_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group_id"] = (
        df["code"].astype(str)         + "_" +
        df["sub_code"].astype(str)     + "_" +
        df["sub_category"].astype(str) + "_" +
        df["horizon"].astype(str)
    )
    return df


def combine_and_sort(train: pd.DataFrame,
                     test:  pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    test  = test.copy()

    train["is_test"] = 0
    test["is_test"]  = 1

    combined = pd.concat([train, test], ignore_index=True)
    combined = combined.sort_values(["group_id", "ts_index"]).reset_index(drop=True)

    print(f"Combined shape after sort: {combined.shape}")
    return combined

def add_lag_features(df: pd.DataFrame,
                     lags: list[int] = [1, 2, 3, 7, 14, 30]) -> pd.DataFrame:
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("group_id")["y_target"].shift(lag)
    print(f"  Lag features added: {[f'lag_{l}' for l in lags]}")
    return df


def add_rolling_features(df: pd.DataFrame,
                         windows: list[int] = [7, 14, 30]) -> pd.DataFrame:
    for w in windows:
        shifted = df.groupby("group_id")["y_target"].shift(1)
        df[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
        df[f"rolling_std_{w}"]  = shifted.rolling(w).std()
    print(f"  Rolling features added for windows: {windows}")
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df["time"]    = df["ts_index"]
    df["time_sq"] = df["ts_index"] ** 2
    print("  Trend features added: time, time_sq")
    return df


def add_category_stats(df: pd.DataFrame) -> pd.DataFrame:
    df["category_mean"] = (
        df.groupby("sub_category")["y_target"]
        .shift(1)
        .expanding()
        .mean()
    )
    df["code_mean"] = (
        df.groupby("code")["y_target"]
        .shift(1)
        .expanding()
        .mean()
    )
    print("  Category stats added: category_mean, code_mean")
    return df


def engineer_features(combined: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features on combined df...")
    combined = add_lag_features(combined)
    combined = add_rolling_features(combined)
    combined = add_trend_features(combined)
    combined = add_category_stats(combined)
    print(f"  Shape after feature engineering: {combined.shape}")
    return combined


def label_encode(train: pd.DataFrame,
                 test:  pd.DataFrame,
                 cols:  list[str] = ["code", "sub_code", "sub_category"]
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Label encoding categorical columns...")
    for col in cols:
        le = LabelEncoder()
        combined_vals = (
            pd.concat([train[col], test[col]])
            .astype(str)
            .unique()
        )
        le.fit(combined_vals)
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))
        print(f"  {col}: {len(combined_vals)} unique values encoded")
    return train, test



def split_and_clean(combined: pd.DataFrame
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Splitting combined df back into train / test...")

    train_features = combined[combined["is_test"] == 0].copy()
    test_features  = combined[combined["is_test"] == 1].copy()

    print(f"  Before dropna — Train: {train_features.shape}  Test: {test_features.shape}")

    train_features = train_features.dropna()

    print(f"  After  dropna — Train: {train_features.shape}  Test: {test_features.shape}")

    # is_test was only a routing flag — remove it from both
    train_features = train_features.drop(columns=["is_test"])
    test_features  = test_features.drop(columns=["is_test"])

    return train_features, test_features

def save(train_features: pd.DataFrame,
         test_features:  pd.DataFrame,
         train_out: str = "train_features_cleaned.parquet",
         test_out:  str = "test_features_processed.parquet") -> None:
    train_features.to_parquet(train_out, index=False)
    test_features.to_parquet(test_out,  index=False)
    print(f"Saved: {train_out}  ({train_features.shape})")
    print(f"Saved: {test_out}   ({test_features.shape})")

def run_pipeline(train_path: str,
                 test_path:  str,
                 train_out:  str = "train_features_cleaned.parquet",
                 test_out:   str = "test_features_processed.parquet") -> None:

    # 1. Load
    train, test = load_data(train_path, test_path)

    # 2. group_id on both before combining
    train = add_group_id(train)
    test  = add_group_id(test)

    # 3. Combine and sort chronologically
    combined = combine_and_sort(train, test)

    # 4. Feature engineering on combined df
    combined = engineer_features(combined)

    # 5. Label encode directly on combined df — avoids dtype mismatch when
    print("Label encoding categorical columns...")
    for col in ["code", "sub_code", "sub_category"]:
        le = LabelEncoder()
        vals = combined[col].astype(str)
        combined[col] = le.fit_transform(vals)
        print(f"  {col}: {combined[col].nunique()} unique values encoded")

    # 6. Split, dropna train only
    train_features, test_features = split_and_clean(combined)

    # 7. Save
    save(train_features, test_features, train_out, test_out)

    print("\nPipeline complete.")
    print(f"  Final train columns ({len(train_features.columns)}): {train_features.columns.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMS 580 data cleaning pipeline")
    parser.add_argument("--train",     default="train.parquet",
                        help="Path to raw train.parquet")
    parser.add_argument("--test",      default="test.parquet",
                        help="Path to raw test.parquet")
    parser.add_argument("--train_out", default="train_features_cleaned.parquet",
                        help="Output path for cleaned train parquet")
    parser.add_argument("--test_out",  default="test_features_processed.parquet",
                        help="Output path for processed test parquet")
    args = parser.parse_args()

    run_pipeline(args.train, args.test, args.train_out, args.test_out)

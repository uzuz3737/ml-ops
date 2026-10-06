import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Column name mapping to standardize UCI feature names
UCI_FEATURE_MAP = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": "default_payment_next_month",
}


def load_raw_dataset(project_root: Path) -> pd.DataFrame:
    """Loads dataset from local files or fallback UCI repo."""
    # 1. Check pre-bundled raw CSV
    raw_csv = project_root / "data" / "raw" / "credit_card_clients.csv"
    if raw_csv.exists():
        logger.info(f"Loading raw dataset from local CSV: {raw_csv}")
        df = pd.read_csv(raw_csv).rename(columns=UCI_FEATURE_MAP)
        return df

    # 2. Check local Excel file
    local_xls = project_root / "default of credit card clients.xls"
    if local_xls.exists():
        logger.info(f"Loading from local Excel: {local_xls}")
        try:
            df = pd.read_excel(local_xls, header=1)
            if "ID" in df.columns:
                df = df.drop(columns=["ID"])
            df = df.rename(columns={"default payment next month": "default_payment_next_month"})
            return df
        except Exception as ex:
            logger.warning(f"Could not read local excel: {ex}")

    # 3. Fallback to UCI repository
    try:
        logger.info("Attempting to fetch Credit Card Default dataset from UCI ML repo (id=350)...")
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=350)
        X = dataset.data.features
        y = dataset.data.targets

        X = X.rename(columns=UCI_FEATURE_MAP)
        y = y.rename(columns=UCI_FEATURE_MAP)

        df = pd.concat([X, y], axis=1)
        if "default_payment_next_month" not in df.columns:
            df = df.rename(columns={df.columns[-1]: "default_payment_next_month"})

        logger.info(f"Successfully fetched dataset from UCI with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Could not fetch from UCI repo: {e}")
        raise FileNotFoundError("No available data source found (local CSV, Excel, or UCI API).")


def ingest_data(config: dict = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ingests data, performs train/validation/test split, and persists datasets."""
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_dir = root / config["data"]["raw_dir"]
    proc_dir = root / config["data"]["processed_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset(root)

    # Standardize column types
    target_col = config["data"]["target_col"]
    df[target_col] = df[target_col].astype(int)

    # Save complete raw data
    raw_csv_path = root / config["data"]["raw_data_file"]
    df.to_csv(raw_csv_path, index=False)
    logger.info(f"Saved raw dataset to {raw_csv_path}")

    # Train / Val / Test split
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    seed = config["data"]["random_state"]

    train_val, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[target_col]
    )

    # Adjust val_size relative to remaining train_val
    adj_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val, test_size=adj_val_size, random_state=seed, stratify=train_val[target_col]
    )

    train_path = root / config["data"]["train_data_file"]
    val_path = root / config["data"]["val_data_file"]
    test_path = root / config["data"]["test_data_file"]

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info(
        f"Data ingestion complete. Splits saved to {proc_dir} -> "
        f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
    )

    return train_df, val_df, test_df


if __name__ == "__main__":
    ingest_data()

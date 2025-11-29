import os
import logging
from .config import LOG_DIR, DATA_PATH
import pandas as pd

# Logging setup
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV data from given path."""
    try:
        df = pd.read_csv(path)
        logger.info(f"Data loaded from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def basic_eda(df: pd.DataFrame) -> None:
    """Print basic EDA info."""
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nDescribe (numeric):\n", df.describe())

    logger.info("Basic EDA completed.")

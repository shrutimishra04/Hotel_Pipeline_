import os
import logging
from .config import LOG_DIR
import pandas as pd
import numpy as np

# Logging setup
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for hotel dataset.
    Creates:
    - total_guests
    - total_nights
    - weekend_heavy
    - lead_time_category
    - avg_price_per_person
    - stay_duration_category
    """
    df = df.copy()
    logger.info("Starting feature engineering.")

    # 1. Total guests
    df["total_guests"] = df["no_of_adults"] + df["no_of_children"]

    # 2. Total nights
    df["total_nights"] = df["no_of_week_nights"] + df["no_of_weekend_nights"]

    # 3. Weekend heavy flag
    df["weekend_heavy"] = (df["no_of_weekend_nights"] > df["no_of_week_nights"]).astype(int)

    # 4. Lead time category
    def lead_cat(x):
        if x <= 30:
            return "short"
        elif x <= 90:
            return "medium"
        else:
            return "long"

    df["lead_time_category"] = df["lead_time"].apply(lead_cat)

    # 5. Average price per person
    df["total_guests_safe"] = df["total_guests"].replace(0, 1)
    df["avg_price_per_person"] = df["avg_price_per_room"] / df["total_guests_safe"]
    df.drop(columns=["total_guests_safe"], inplace=True)

    # 6. Stay duration category
    def stay_cat(x):
        if x <= 2:
            return "short_stay"
        elif x <= 5:
            return "medium_stay"
        else:
            return "long_stay"

    df["stay_duration_category"] = df["total_nights"].apply(stay_cat)

    logger.info("Feature engineering completed.")
    return df

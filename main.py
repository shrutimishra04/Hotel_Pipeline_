import os
import logging

from src.data_loader import load_data, basic_eda
from src.feature_engineering import engineer_features
from src.preprocess import clean_data, treat_outliers_iqr, split_data
from src.train import (
    train_and_compare_models,
    tune_random_forest,
    save_model,
)
from src.evaluate import get_feature_importance, evaluate_model
from src.config import LOG_DIR

# Logging setup for main
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # 1. Load data
    df_raw = load_data()

    # 2. Basic EDA
    basic_eda(df_raw)

    # 3. Feature engineering
    df_fe = engineer_features(df_raw)

    # 4. Cleaning + Outliers
    df_clean = clean_data(df_fe)
    df_clean = treat_outliers_iqr(df_clean)

    # 5. Split
    X_train, X_test, y_train, y_test = split_data(df_clean)

    # 6. Train base models
    best_name, best_pipeline, results, (num_cols, cat_cols) = train_and_compare_models(
        X_train, y_train, X_test, y_test, df_clean
    )

    print("\nBase Models performance:")
    for name, metrics in results.items():
        print(name, "=>", metrics)

    # 7. Tune Random Forest
    tuned_rf = tune_random_forest(X_train, y_train, df_clean)

    # 8. Evaluate tuned model
    y_pred_tuned = tuned_rf.predict(X_test)
    y_prob_tuned = tuned_rf.predict_proba(X_test)[:, 1]
    tuned_metrics = evaluate_model(y_test, y_pred_tuned, y_prob_tuned)
    print("\nTuned RF metrics:", tuned_metrics)

    # 9. Choose final model (best F1)
    final_model = tuned_rf
    if results[best_name]["f1"] > tuned_metrics["f1"]:
        print("\nUsing best base model as final (higher F1).")
        final_model = best_pipeline
    else:
        print("\nUsing tuned Random Forest as final model.")

    # 10. Feature importance (if available)
    get_feature_importance(final_model, num_cols, cat_cols)

    # 11. Save model
    save_model(final_model, filename="hotel_cancellation_model.joblib")

if __name__ == "__main__":
    main()

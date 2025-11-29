import os
import logging
from .config import LOG_DIR, RANDOM_STATE, MODEL_DIR
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from .preprocess import get_preprocessor
from .evaluate import evaluate_model

import joblib

# Logging setup
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def build_pipeline(preprocessor, model):
    """
    Build imblearn pipeline: preprocess -> SMOTE -> classifier.
    """
    pipe = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", model),
        ]
    )
    return pipe

def train_and_compare_models(X_train, y_train, X_test, y_test, df_full):
    """
    Train Logistic Regression, Random Forest, Gradient Boosting.
    Return best model and metrics.
    """
    preprocessor, num_cols, cat_cols = get_preprocessor(df_full)

    models = {
        "log_reg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results: Dict[str, Dict] = {}
    best_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, model in models.items():
        logger.info(f"Training model: {name}")
        pipe = build_pipeline(preprocessor, model)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_prob)
        results[name] = metrics

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_pipeline = pipe

    logger.info(f"Best base model: {best_name} with F1={best_f1:.4f}")
    return best_name, best_pipeline, results, (num_cols, cat_cols)

def tune_random_forest(X_train, y_train, df_full):
    """
    Hyperparameter tuning for Random Forest using RandomizedSearchCV.
    """
    logger.info("Starting Random Forest hyperparameter tuning.")
    preprocessor, _, _ = get_preprocessor(df_full)

    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    pipe = build_pipeline(preprocessor, base_model)

    param_distributions = {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    logger.info(f"Best RF params: {search.best_params_}")
    logger.info(f"Best RF F1 (CV): {search.best_score_:.4f}")

    return search.best_estimator_

def save_model(model, filename: str = "hotel_cancellation_model.joblib"):
    """
    Save model using joblib to models folder.
    """
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    logger.info(f"Model saved at {path}")
    print(f"Model saved at {path}")

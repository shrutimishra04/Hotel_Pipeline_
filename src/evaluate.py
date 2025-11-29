import os
import logging
from .config import LOG_DIR
from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Logging setup
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, y_prob) -> Dict:
    """
    Print confusion matrix, classification report, and return metrics dict.
    """
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    print("\nMetrics:", metrics)
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

def get_feature_importance(trained_pipeline, num_cols, cat_cols):
    """
    Get feature importances from tree-based model if available.
    """
    clf = trained_pipeline.named_steps.get("clf")
    if clf is None or not hasattr(clf, "feature_importances_"):
        print("Feature importance not available for this model.")
        logger.warning("Classifier has no feature_importances_.")
        return None

    preprocess = trained_pipeline.named_steps["preprocess"]
    ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols)

    all_feature_names = list(num_cols) + list(cat_feature_names)
    importances = clf.feature_importances_

    importance_pairs = sorted(
        zip(all_feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\nTop 15 Feature Importances:")
    for name, imp in importance_pairs[:15]:
        print(f"{name}: {imp:.4f}")

    return importance_pairs

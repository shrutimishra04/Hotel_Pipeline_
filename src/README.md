# Hotel Cancellation Prediction Project

This project predicts whether a hotel booking will be **canceled or not**
using the "Hotel Reservations" dataset from Kaggle.

## Pipeline Steps

1. Exploratory Data Analysis (EDA)
2. Data Cleaning
3. Feature Engineering
4. Outlier Detection & Treatment
5. Encoding & Scaling (via ColumnTransformer)
6. Handling Class Imbalance (SMOTE)
7. Model Training (Logistic Regression, Random Forest, Gradient Boosting)
8. Hyperparameter Tuning (RandomizedSearchCV on Random Forest)
9. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
10. CI/CD using GitHub Actions
11. Model Saving with joblib

## Project Structure

See the folder tree in the report or repository.

## Run Locally

```bash
pip install -r requirements.txt
python main.py

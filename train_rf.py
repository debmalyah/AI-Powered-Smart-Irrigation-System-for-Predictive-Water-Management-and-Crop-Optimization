import json
import mlflow
import mlflow.sklearn
import joblib
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import get_preprocessed_data

ARTIFACT_PATH = "rf_irrigation_model.joblib"
METRICS_PATH = "rf_metrics.json"

# Same search budget (n_iter=20, 3-fold stratified CV, accuracy scoring) as
# train_gb.py's RandomizedSearchCV -- deliberately matched so the RF-vs-GB
# comparison in select_best_model.py reflects a real difference between the
# algorithms, not one of them being tuned and the other left at defaults.
RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 400),
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 20),
    "max_features": ["sqrt", "log2", None],
}


def train_rf_with_mlflow():
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     _, scaler, feature_columns,
     soil_moisture_max, rainfall_max) = get_preprocessed_data()

    mlflow.set_experiment("Irrigation_Prediction_Engine")

    with mlflow.start_run(run_name="Random_Forest_Tuned"):
        cv_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        print("Executing RandomizedSearchCV for Random Forest...")
        rf_random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=RF_PARAM_DISTRIBUTIONS,
            n_iter=20,
            cv=cv_search,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
        )
        rf_random_search.fit(X_train, y_train)
        model = rf_random_search.best_estimator_

        mlflow.log_params(rf_random_search.best_params_)

        # Validation metrics: checked before test is touched at all.
        y_val_pred = model.predict(X_val)
        val_metrics = {
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred, average="weighted"),
            "val_recall": recall_score(y_val, y_val_pred, average="weighted"),
            "val_f1_score": f1_score(y_val, y_val_pred, average="weighted")
        }
        mlflow.log_metrics(val_metrics)

        # Test metrics: the final, one-shot reported numbers.
        y_pred = model.predict(X_test)
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, average="weighted"),
            "test_recall": recall_score(y_test, y_pred, average="weighted"),
            "test_f1_score": f1_score(y_test, y_pred, average="weighted")
        }
        mlflow.log_metrics(test_metrics)

        # Log and version model artifact in MLflow Registry
        mlflow.sklearn.log_model(model, name="model", registered_model_name="RF_Irrigation_Model")

        # Save local joblib artifact for FastAPI
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "soil_moisture_max": soil_moisture_max,
            "rainfall_max": rainfall_max,
        }, ARTIFACT_PATH)

        # Same flat schema as baseline_metrics.json / gb_metrics.json --
        # select_best_model.py reads all three interchangeably.
        metrics_out = {
            "model_name": "Random_Forest",
            **val_metrics,
            **test_metrics,
            "artifact_path": ARTIFACT_PATH,
            "best_params": rf_random_search.best_params_,
        }
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics_out, f, indent=2)

        print(f"Best params: {rf_random_search.best_params_}")
        print(f"Validation metrics: {val_metrics}")
        print(f"Test metrics: {test_metrics}")
        print("Model, metrics, and params successfully logged to MLflow.")
        print(f"Comparable metrics saved to '{METRICS_PATH}'.")


if __name__ == "__main__":
    train_rf_with_mlflow()
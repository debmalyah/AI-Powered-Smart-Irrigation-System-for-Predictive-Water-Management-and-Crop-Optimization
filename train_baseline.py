import json
import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from data_preprocessing import get_preprocessed_data, TARGET_ORDER


def _evaluate(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y, y_pred, average="weighted", zero_division=0),
    }


def train_baseline():
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     _, scaler, feature_columns,
     soil_moisture_max, rainfall_max) = get_preprocessed_data()

    # Same experiment as train_rf.py / train_gb.py so all three (plus the
    # dummy floor, logged as a param/tag below) show up side by side in the
    # MLflow UI -- previously this script wasn't instrumented at all, so the
    # baseline run was invisible next to RF/GB in MLflow.
    mlflow.set_experiment("Irrigation_Prediction_Engine")

    with mlflow.start_run(run_name="Baseline_LogisticRegression"):
        # Trivial sanity-check floor: always predicts the majority class. Any
        # real model that doesn't clearly beat this isn't learning anything.
        # Logged as metrics (prefixed dummy_) rather than left out of MLflow,
        # so the floor is visible alongside the real baseline run.
        dummy = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy.fit(X_train, y_train)
        dummy_val = _evaluate(dummy, X_val, y_val)
        dummy_test = _evaluate(dummy, X_test, y_test)
        mlflow.log_metrics({f"dummy_val_{k}": v for k, v in dummy_val.items()})
        mlflow.log_metrics({f"dummy_test_{k}": v for k, v in dummy_test.items()})

        # The actual baseline: plain multinomial logistic regression, no tuning,
        # no SMOTE. This is the bar RF/GB need to clear to justify their extra
        # complexity -- if they don't beat this by a healthy margin, the added
        # complexity (and tuning time) isn't paying for itself.
        # No hyperparameter search here by design, but the fixed settings are
        # still logged as params so the run is self-documenting in MLflow.
        logreg_params = {"max_iter": 1000, "random_state": 42}
        mlflow.log_params(logreg_params)
        mlflow.set_tag("tuned", "false")
        mlflow.set_tag("model_family", "logistic_regression")

        logreg = LogisticRegression(**logreg_params)
        logreg.fit(X_train, y_train)
        logreg_val = _evaluate(logreg, X_val, y_val)
        logreg_test = _evaluate(logreg, X_test, y_test)

        val_metrics_mlflow = {f"val_{k}": v for k, v in logreg_val.items()}
        test_metrics_mlflow = {f"test_{k}": v for k, v in logreg_test.items()}
        mlflow.log_metrics(val_metrics_mlflow)
        mlflow.log_metrics(test_metrics_mlflow)

        print("\n" + "=" * 70)
        print("BASELINE: Dummy (most-frequent class) -- sanity floor only")
        print("=" * 70)
        print(f"Validation: {dummy_val}")
        print(f"Test:       {dummy_test}")

        print("\n" + "=" * 70)
        print("BASELINE: Logistic Regression")
        print("=" * 70)
        print(f"Validation: {logreg_val}")
        print(f"Test:       {logreg_test}")
        print(classification_report(y_test, logreg.predict(X_test), target_names=TARGET_ORDER))

        # Log and version model artifact in MLflow Registry -- same treatment
        # RF and GB get, so the baseline is a real, comparable entry in the
        # registry rather than a joblib file MLflow never saw.
        mlflow.sklearn.log_model(
            logreg, name="model", registered_model_name="Baseline_Irrigation_Model"
        )

        artifact_path = "baseline_irrigation_model.joblib"
        joblib.dump(
            {
                "model": logreg, "scaler": scaler, "feature_columns": feature_columns,
                "soil_moisture_max": soil_moisture_max, "rainfall_max": rainfall_max,
            },
            artifact_path
        )

        # Same flat schema as rf_metrics.json / gb_metrics.json on purpose --
        # select_best_model.py reads all three interchangeably.
        metrics = {
            "model_name": "Baseline_LogisticRegression",
            "dummy_floor": {"val": dummy_val, "test": dummy_test},
            "val_accuracy": logreg_val["accuracy"],
            "val_precision": logreg_val["precision"],
            "val_recall": logreg_val["recall"],
            "val_f1_score": logreg_val["f1_score"],
            "test_accuracy": logreg_test["accuracy"],
            "test_precision": logreg_test["precision"],
            "test_recall": logreg_test["recall"],
            "test_f1_score": logreg_test["f1_score"],
            "artifact_path": artifact_path,
        }
        with open("baseline_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nBaseline model saved to '{artifact_path}'.")
        print("Baseline metrics saved to 'baseline_metrics.json'.")
        print("Model, metrics, and params successfully logged to MLflow.")

        return metrics


if __name__ == "__main__":
    train_baseline()
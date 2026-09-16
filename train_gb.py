import json
import mlflow
import mlflow.sklearn
import joblib
from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from data_preprocessing import get_preprocessed_data, TARGET_ORDER

ARTIFACT_PATH = "gb_irrigation_model.joblib"
METRICS_PATH = "gb_metrics.json"

def run_gradient_boosting():
    (X_train_scaled, X_val_scaled, X_test_scaled,
     y_train, y_val, y_test,
     cv, scaler, feature_columns,
     soil_moisture_max, rainfall_max) = get_preprocessed_data()

    # Same MLflow experiment as train_rf.py so both models' runs show up
    # side by side in the MLflow UI.
    mlflow.set_experiment("Irrigation_Prediction_Engine")

    with mlflow.start_run(run_name="Gradient_Boosting_Tuned"):
        smote_gb_pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("gb", HistGradientBoostingClassifier(
                random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10
            )),
        ])

        gb_param_distributions = {
            "gb__max_iter": randint(100, 300),
            "gb__learning_rate": uniform(0.02, 0.28),
            "gb__max_depth": [None, 3, 5, 7, 10],
            "gb__max_leaf_nodes": randint(15, 63),
            "gb__min_samples_leaf": randint(10, 40),
        }

        cv_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        print("Executing RandomizedSearchCV for Gradient Boosting...")
        gb_random_search = RandomizedSearchCV(
            estimator=smote_gb_pipeline,
            param_distributions=gb_param_distributions,
            n_iter=20,
            cv=cv_search,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
        )
        gb_random_search.fit(X_train_scaled, y_train)

        best_gb_model = gb_random_search.best_estimator_

        # MLflow only accepts plain scalar param values, so stringify anything
        # else (e.g. None inside gb__max_depth) before logging.
        loggable_params = {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                            for k, v in gb_random_search.best_params_.items()}
        mlflow.log_params(loggable_params)

        # Validation set: sanity-check the tuned model on data it never saw during
        # CV-based tuning, before touching the test set at all.
        y_val_pred = best_gb_model.predict(X_val_scaled)
        val_metrics = {
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred, average="weighted", zero_division=0),
            "val_recall": recall_score(y_val, y_val_pred, average="weighted", zero_division=0),
            "val_f1_score": f1_score(y_val, y_val_pred, average="weighted", zero_division=0),
        }
        mlflow.log_metrics(val_metrics)
        print(f"\nValidation Accuracy: {val_metrics['val_accuracy']:.4f}\n")
        print(classification_report(y_val, y_val_pred, target_names=TARGET_ORDER))

        # Test set: touched exactly once, for the final reported number.
        y_pred = best_gb_model.predict(X_test_scaled)
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "test_recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "test_f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        mlflow.log_metrics(test_metrics)

        print(f"\nTuned Gradient Boosting Test Accuracy: {test_metrics['test_accuracy']:.4f}\n")
        print(classification_report(y_test, y_pred, target_names=TARGET_ORDER))

        # Log and version model artifact in MLflow Registry.
        # mlflow's sklearn flavor serializes with skops (safer than pickle),
        # which by default refuses to trust anything outside scikit-learn's
        # own types. Our estimator is an imblearn Pipeline wrapping SMOTE,
        # so both have to be explicitly declared trusted -- SMOTE only runs
        # during .fit() (imblearn skips resampling steps at .predict() time),
        # so this doesn't change what the logged model actually does at
        # inference, it's purely a "yes, I wrote/trust this pipeline" ack.
        mlflow.sklearn.log_model(
            best_gb_model, name="model", registered_model_name="GB_Irrigation_Model",
            skops_trusted_types=[
                "imblearn.over_sampling._smote.base.SMOTE",
                "imblearn.pipeline.Pipeline",
            ],
        )

        # Save model and preprocessor metadata
        joblib.dump(
            {
                "model": best_gb_model, "scaler": scaler, "feature_columns": feature_columns,
                "soil_moisture_max": soil_moisture_max, "rainfall_max": rainfall_max,
            },
            ARTIFACT_PATH
        )
        print(f"Model successfully saved to '{ARTIFACT_PATH}'.")

        # Same flat schema as baseline_metrics.json / rf_metrics.json --
        # select_best_model.py reads all three interchangeably.
        metrics_out = {
            "model_name": "Gradient_Boosting",
            **val_metrics,
            **test_metrics,
            "artifact_path": ARTIFACT_PATH,
            "best_params": loggable_params,
        }
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics_out, f, indent=2)
        print(f"Comparable metrics saved to '{METRICS_PATH}'.")
        print("Model, metrics, and params successfully logged to MLflow.")

if __name__ == "__main__":
    run_gradient_boosting()
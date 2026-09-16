import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import get_preprocessed_data


def train_rf_with_mlflow():
    X_train, X_test, y_train, y_test, _, scaler, feature_columns = get_preprocessed_data()

    mlflow.set_experiment("Irrigation_Prediction_Engine")

    with mlflow.start_run(run_name="Random_Forest_Tuned"):
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted")
        }
        mlflow.log_metrics(metrics)

        # Log and version model artifact in MLflow Registry
        mlflow.sklearn.log_model(model, name="model", registered_model_name="RF_Irrigation_Model")

        # Save local joblib artifact for FastAPI
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns
        }, "rf_irrigation_model.joblib")

        print("Model, metrics, and params successfully logged to MLflow.")


if __name__ == "__main__":
    train_rf_with_mlflow()
import joblib
from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from data_preprocessing import get_preprocessed_data, TARGET_ORDER

def run_gradient_boosting():
    (X_train_scaled, X_test_scaled, y_train, y_test,
     cv, scaler, feature_columns) = get_preprocessed_data()

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
    y_pred = best_gb_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTuned Gradient Boosting Test Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=TARGET_ORDER))

    # Save model and preprocessor metadata
    joblib.dump(
        {"model": best_gb_model, "scaler": scaler, "feature_columns": feature_columns},
        "gb_irrigation_model.joblib"
    )
    print("Model successfully saved to 'gb_irrigation_model.joblib'.")

if __name__ == "__main__":
    run_gradient_boosting()
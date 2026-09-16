"""
LSTM experiment for irrigation need classification.

IMPORTANT CONTEXT -- READ BEFORE RE-ENABLING THIS IN select_best_model.py:

irrigation_prediction.csv has no Timestamp / Field_ID / any sequence key
(confirmed in data_eda.py: "the dataset has no timestamp/sequence column").
Each row is an independent observation, not a step in a time series.

The original version of this script built "sequences" by sliding a window
over rows AFTER they had already been through a stratified train_test_split
(which shuffles rows). That means each 5-row "window" was 5 unrelated,
randomly-shuffled records glued together -- there is no real temporal
relationship for the LSTM to learn from. That is almost certainly why
accuracy collapsed to ~44% (barely above the 33% floor for 3 balanced
classes): the model was being asked to find sequential signal in noise.

This version:
  1. Fixes the crash: get_preprocessed_data() now returns 11 values
     (train/val/test), not the old 7.
  2. Builds sliding windows SEPARATELY within each split (train windows only
     from train rows, val windows only from val rows, test windows only from
     test rows) so windows never straddle a split boundary. This avoids one
     leakage bug, but does NOT fix the underlying issue -- windows within a
     split are still built from shuffled rows, since the split upstream is
     stratified, not order-preserving.
  3. Stops EarlyStopping from monitoring the TEST set (it should never see
     test data before final evaluation) -- it now monitors the VALIDATION
     set, matching how train_rf.py / train_gb.py / train_baseline.py do it.
  4. Saves a metrics JSON in the same flat schema as the other models,
     WITH an explicit caveat field, so it can be cited in the model
     comparison report -- but is deliberately NOT added to
     select_best_model.py's CANDIDATES dict. A low score here is evidence
     for "this dataset isn't sequential," not a model to ship.

Recommendation: keep this as a documented negative result in the Milestone 2
report (justifying why baseline/RF/GB -- not LSTM -- is the right model
family for this problem), rather than trying to tune it further.
"""
import os

os.environ["KERAS_BACKEND"] = "torch"  # Must be set before importing Keras

import json
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from data_preprocessing import TARGET_ORDER, get_preprocessed_data

ARTIFACT_PATH = "lstm_irrigation_model.keras"
METRICS_PATH = "lstm_metrics.json"
WINDOW_SIZE = 5
STEP_SIZE = 1


def create_sliding_windows(X, y, window_size=5, step_size=1):
    """Builds 3D sliding-window sequences (samples, window_size, features) and
    aligns target labels. Caller is responsible for only passing rows from a
    single split, so windows never straddle a train/val/test boundary.

    NOTE: this treats row order as meaningful. Since there is no real
    timestamp/sequence key in this dataset, "order" here just means
    whatever order the rows happen to be in after preprocessing -- see the
    module docstring. Windows are a diagnostic, not a validated temporal
    model.
    """
    X_seq, y_seq = [], []
    for i in range(0, len(X) - window_size, step_size):
        X_seq.append(X[i: i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)


def _evaluate(model, X_seq, y_seq):
    y_pred_proba = model.predict(X_seq, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    return {
        "accuracy": accuracy_score(y_seq, y_pred),
        "precision": precision_score(y_seq, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_seq, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_seq, y_pred, average="weighted", zero_division=0),
    }, y_pred


def train_lstm_model():
    # 1. Fetch preprocessed data -- 11 values now that a validation split exists.
    (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        _cv, scaler, feature_columns,
        soil_moisture_max, rainfall_max,
    ) = get_preprocessed_data()

    y_train_arr = y_train.values
    y_val_arr = y_val.values
    y_test_arr = y_test.values

    # 2. Build sliding windows PER SPLIT (never combine splits before windowing --
    # that would leak rows across the train/val/test boundary regardless of the
    # ordering issue above).
    X_train_seq, y_train_seq = create_sliding_windows(X_train_scaled, y_train_arr, WINDOW_SIZE, STEP_SIZE)
    X_val_seq, y_val_seq = create_sliding_windows(X_val_scaled, y_val_arr, WINDOW_SIZE, STEP_SIZE)
    X_test_seq, y_test_seq = create_sliding_windows(X_test_scaled, y_test_arr, WINDOW_SIZE, STEP_SIZE)

    print(f"Sequence counts -> train: {len(X_train_seq)}, val: {len(X_val_seq)}, test: {len(X_test_seq)}")

    # 3. Class weights from the TRAIN windows only.
    classes = np.unique(y_train_seq)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_seq)
    class_weight_dict = dict(zip(classes.tolist(), class_weights.tolist()))

    y_train_cat = to_categorical(y_train_seq, num_classes=len(TARGET_ORDER))
    y_val_cat = to_categorical(y_val_seq, num_classes=len(TARGET_ORDER))

    # 4. Build LSTM architecture.
    num_features = X_train_seq.shape[2]
    model = Sequential([
        Input(shape=(WINDOW_SIZE, num_features)),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(len(TARGET_ORDER), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Monitors VALIDATION loss, not test -- test stays untouched until final eval,
    # matching the other training scripts in this pipeline.
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    print(f"Training LSTM across {len(X_train_seq)} sequence windows...")
    model.fit(
        X_train_seq, y_train_cat,
        validation_data=(X_val_seq, y_val_cat),
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # 5. Validation metrics (checked before test is touched at all).
    val_metrics, _ = _evaluate(model, X_val_seq, y_val_seq)
    print(f"\nValidation Accuracy: {val_metrics['accuracy']:.4f}")

    # 6. Test metrics -- touched exactly once, for the final reported number.
    test_metrics, y_test_pred = _evaluate(model, X_test_seq, y_test_seq)
    print(f"\nLSTM Test Accuracy: {test_metrics['accuracy']:.4f}\n")
    print(classification_report(y_test_seq, y_test_pred, target_names=TARGET_ORDER, zero_division=0))

    model.save(ARTIFACT_PATH)
    print(f"Model saved to '{ARTIFACT_PATH}' (diagnostic artifact -- not wired into select_best_model.py).")

    # Same flat schema as baseline/rf/gb metrics files, plus an explicit caveat
    # so anyone reading this later (or select_best_model.py, if it's ever
    # pointed at this file) understands why a low score here isn't a bug.
    metrics_out = {
        "model_name": "LSTM_SlidingWindow",
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1_score": val_metrics["f1_score"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1_score": test_metrics["f1_score"],
        "artifact_path": ARTIFACT_PATH,
        "caveat": (
            "irrigation_prediction.csv has no timestamp/sequence key, so the "
            "windows this model was trained on are not genuinely sequential. "
            "This result is kept as a documented negative finding -- it "
            "justifies treating the problem as tabular classification "
            "(baseline/RF/GB), not as evidence this architecture needs more "
            "tuning. Do not add this model to select_best_model.py's "
            "CANDIDATES."
        ),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved to '{METRICS_PATH}' (for the report, not for model selection).")

    return metrics_out


if __name__ == "__main__":
    train_lstm_model()
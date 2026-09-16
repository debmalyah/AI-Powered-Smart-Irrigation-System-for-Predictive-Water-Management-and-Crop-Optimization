import os

os.environ["KERAS_BACKEND"] = "torch"  # Must be set before importing Keras

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from data_preprocessing import TARGET_ORDER, get_preprocessed_data


def create_sliding_windows(X, y, window_size=5, step_size=1):
  """Constructs 3D sliding window sequences (samples, window_size, features)

  and aligns target labels.

  Parameters:
      X (np.ndarray): Feature matrix of shape (N, num_features).
      y (np.ndarray): Target array of shape (N,).
      window_size (int): Number of historical timesteps per sequence.
      step_size (int): Stride/step size for moving the window.

  Returns:
      X_seq (np.ndarray): 3D array of shape (num_windows, window_size,
      num_features).
      y_seq (np.ndarray): 1D array of target values following each window.
  """
  X_seq, y_seq = [], []
  for i in range(0, len(X) - window_size, step_size):
    X_seq.append(X[i : i + window_size])
    y_seq.append(y[i + window_size])  # Target at timestep t + window_size

  return np.array(X_seq), np.array(y_seq)


def train_lstm_model():
  # 1. Fetch preprocessed data
  (
      X_train_scaled,
      X_test_scaled,
      y_train,
      y_test,
      _,
      scaler,
      feature_columns,
  ) = get_preprocessed_data()

  # Reconstruct continuous full dataset to maintain unbroken time sequence
  X_full = np.vstack((X_train_scaled, X_test_scaled))
  y_full = np.concatenate((y_train.values, y_test.values))

  # 2. Apply Sliding Window
  WINDOW_SIZE = 5  # Number of timesteps per window
  STEP_SIZE = 1  # Stride length

  X_seq, y_seq = create_sliding_windows(
      X_full, y_full, window_size=WINDOW_SIZE, step_size=STEP_SIZE
  )

  # 3. Chronological Train-Test Split (Preserves time order, no future leakage)
  split_idx = int(len(X_seq) * 0.8)
  X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
  y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

  # 4. Compute Class Weights for Imbalanced Targets
  classes = np.unique(y_train_seq)
  class_weights = compute_class_weight(
      class_weight="balanced", classes=classes, y=y_train_seq
  )
  class_weight_dict = dict(zip(classes, class_weights))

  # One-hot encode targets for categorical crossentropy
  y_train_cat = to_categorical(y_train_seq, num_classes=len(TARGET_ORDER))
  y_test_cat = to_categorical(y_test_seq, num_classes=len(TARGET_ORDER))

  # 5. Build LSTM Architecture
  num_features = X_seq.shape[2]
  model = Sequential([
      Input(shape=(WINDOW_SIZE, num_features)),
      LSTM(64, return_sequences=False),
      BatchNormalization(),
      Dropout(0.3),
      Dense(32, activation="relu"),
      Dense(len(TARGET_ORDER), activation="softmax"),
  ])

  model.compile(
      optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
  )

  callbacks = [
      EarlyStopping(
          monitor="val_loss", patience=10, restore_best_weights=True
      ),
      ReduceLROnPlateau(
          monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
      ),
  ]

  # 6. Model Training
  print(f"Training LSTM Model across {len(X_train_seq)} sequence windows...")
  model.fit(
      X_train_seq,
      y_train_cat,
      validation_data=(X_test_seq, y_test_cat),
      epochs=50,
      batch_size=32,
      class_weight=class_weight_dict,
      callbacks=callbacks,
      verbose=1,
  )

  # 7. Evaluation & Export
  loss, accuracy = model.evaluate(X_test_seq, y_test_cat, verbose=0)
  print(f"\nLSTM Test Accuracy: {accuracy:.4f}")

  model.save("lstm_irrigation_model.keras")
  print("Model successfully saved to 'lstm_irrigation_model.keras'.")


if __name__ == "__main__":
  train_lstm_model()
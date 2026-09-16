import json
import shutil
import subprocess
import sys
from pathlib import Path

# name -> (metrics file it produces, script that produces it)
CANDIDATES = {
    "Baseline (Logistic Regression)": ("baseline_metrics.json", "train_baseline.py"),
    "Random Forest": ("rf_metrics.json", "train_rf.py"),
    "Gradient Boosting": ("gb_metrics.json", "train_gb.py"),
}

# Weighted F1 (not accuracy) is the selection metric: it's more robust than
# accuracy when the three Irrigation_Need classes aren't perfectly balanced,
# and it's computed consistently across all three candidates.
SELECTION_METRIC = "test_f1_score"

BEST_MODEL_PATH = "best_irrigation_model.joblib"
SUMMARY_PATH = "model_selection_summary.json"


def _load_or_train(name: str, metrics_file: str, train_script: str) -> dict:
    path = Path(metrics_file)
    if not path.exists():
        print(f"[{name}] '{metrics_file}' not found -- running {train_script} first...")
        result = subprocess.run([sys.executable, train_script])
        if result.returncode != 0:
            raise RuntimeError(f"{train_script} failed; cannot include {name} in the comparison.")
    with open(path) as f:
        return json.load(f)


def select_best_model():
    all_metrics = {}
    for name, (metrics_file, train_script) in CANDIDATES.items():
        all_metrics[name] = _load_or_train(name, metrics_file, train_script)

    print("\n" + "=" * 78)
    print(f"{'Model':<32}{'Val F1':>10}{'Val Acc':>10}{'Test F1':>10}{'Test Acc':>10}")
    print("=" * 78)
    for name, m in all_metrics.items():
        print(f"{name:<32}{m['val_f1_score']:>10.4f}{m['val_accuracy']:>10.4f}"
              f"{m['test_f1_score']:>10.4f}{m['test_accuracy']:>10.4f}")

    best_name = max(all_metrics, key=lambda n: all_metrics[n][SELECTION_METRIC])
    best_metrics = all_metrics[best_name]

    print("\n" + "=" * 78)
    print(f"SELECTED MODEL: {best_name}  ({SELECTION_METRIC} = {best_metrics[SELECTION_METRIC]:.4f})")
    print("=" * 78)

    baseline_f1 = all_metrics["Baseline (Logistic Regression)"][SELECTION_METRIC]
    if best_name != "Baseline (Logistic Regression)":
        lift = best_metrics[SELECTION_METRIC] - baseline_f1
        print(f"Lift over baseline: {lift:+.4f} {SELECTION_METRIC}")
        if lift < 0.02:
            print("NOTE: the winning model barely beats the baseline (<0.02 F1). "
                  "Worth double-checking feature engineering / tuning before shipping it.")

    src = Path(best_metrics["artifact_path"])
    if not src.exists():
        raise FileNotFoundError(
            f"'{src}' referenced by {best_name}'s metrics file doesn't exist. "
            f"Re-run {CANDIDATES[best_name][1]} and try again."
        )
    shutil.copy(src, BEST_MODEL_PATH)

    summary = {
        "selected_model": best_name,
        "selection_metric": SELECTION_METRIC,
        "all_models": all_metrics,
        "artifact_path": BEST_MODEL_PATH,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCopied winning artifact to '{BEST_MODEL_PATH}'.")
    print(f"Full comparison saved to '{SUMMARY_PATH}'.")
    print("\nmain_api.py loads 'best_irrigation_model.joblib' -- rerun this script "
          "whenever you retrain, and the API picks up whichever model wins.")

    return summary


if __name__ == "__main__":
    select_best_model()

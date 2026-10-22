"""Train a Random Forest to predict whether the next H bars are range-bound.

Uses TimeSeriesSplit (NOT k-fold) so that every train fold is strictly
in the past relative to its test fold, with a gap of H bars between them:
each label looks H bars ahead, so without the gap the last training labels
would describe prices inside the test fold (López de Prado's "purging"). Reports precision, recall and F1
on the minority class instead of accuracy, because regime labels are
imbalanced (range-bound bars are rarer than trending bars).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CVResult:
    fold: int
    precision: float
    recall: float
    f1: float
    support_pos: int
    confusion: np.ndarray


@dataclass
class TrainResult:
    cv_results: List[CVResult]
    final_model: RandomForestClassifier
    feature_names: List[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fold": r.fold,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "support_pos": r.support_pos,
                }
                for r in self.cv_results
            ]
        )


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    gap: int = 8,
    n_estimators: int = 200,
    max_depth: int = 8,
    random_state: int = 42,
) -> TrainResult:
    """Run purged TimeSeriesSplit CV on a Random Forest classifier.

    ``X`` and ``y`` should contain training data only. The final model is
    fitted on all of ``X``, so passing hold-out rows here would leak them.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_results: List[CVResult] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # If a fold has only one class in y_train, skip cleanly.
        if y_train.nunique() < 2:
            print(f"  Fold {fold}: only one class present, skipping.")
            continue

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fold_results.append(
            CVResult(
                fold=fold,
                precision=float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
                recall=float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
                f1=float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
                support_pos=int((y_test == 1).sum()),
                confusion=confusion_matrix(y_test, y_pred, labels=[0, 1]),
            )
        )
        print(
            f"  Fold {fold}: precision={fold_results[-1].precision:.3f} "
            f"recall={fold_results[-1].recall:.3f} f1={fold_results[-1].f1:.3f} "
            f"support+={fold_results[-1].support_pos}"
        )

    # Final model trained on everything passed in (the training period only).
    final_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    ).fit(X, y)

    return TrainResult(cv_results=fold_results, final_model=final_clf, feature_names=list(X.columns))


def save_model(model: RandomForestClassifier, path: Path = None) -> Path:
    path = path or (MODELS_DIR / "regime_classifier.pkl")
    joblib.dump(model, path)
    print(f"Saved model -> {path}")
    return path


def load_model(path: Path = None) -> RandomForestClassifier:
    path = path or (MODELS_DIR / "regime_classifier.pkl")
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(result: TrainResult, save_path: Path = None) -> Path:
    """Sum confusion matrices across folds and plot."""
    save_path = save_path or (REPORTS_DIR / "confusion_matrix.png")
    cm_total = np.sum([r.confusion for r in result.cv_results], axis=0)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_total, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Trending", "Range-bound"])
    ax.set_yticklabels(["Trending", "Range-bound"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (summed across CV folds)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm_total[i, j])), ha="center", va="center",
                    color="white" if cm_total[i, j] > cm_total.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot -> {save_path}")
    return save_path


def plot_feature_importance(result: TrainResult, save_path: Path = None) -> Path:
    save_path = save_path or (REPORTS_DIR / "feature_importance.png")
    importances = pd.Series(
        result.final_model.feature_importances_,
        index=result.feature_names,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importances.index, importances.values, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    for i, v in enumerate(importances.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot -> {save_path}")
    return save_path

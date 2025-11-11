# trainer/trainer.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset.data import DataLoader
from model.model import GoldPredictor

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, classification_report, confusion_matrix
)

class Trainer:
    """Handles training, evaluation, and visualization for the Gold Predictor."""

    def __init__(self):
        self.data_loader = DataLoader()

    def run(self):
        df = self.data_loader.fetch_data()

        features = [
            "Open", "High", "Low", "Close", "Volume",
            "MA20", "MA200", "Momentum_10", "Momentum_50", "ROC_4w",
            "Volatility", "DollarReturn", "YieldChange", "RSI",
            "Gold_to_Dollar", "Gold_to_Yield", "Gold_to_SPX",
            "Lag1", "Lag5", "Vol_5", "RSI_Change", "DollarLag", "YieldLag"
        ]
        X, y = df[features], df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=102, shuffle=True)
        ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
        model = GoldPredictor(class_ratio=ratio)
        model.train(X_train, y_train, X_test, y_test)

        y_proba = model.predict_proba(X_test)
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
        best_thresh = thresholds[np.argmax(f1_scores)]
        print(f"Optimal threshold: {best_thresh:.3f}")
        y_pred = (y_proba > best_thresh).astype(int)

        # --- Evaluation ---
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = auc(recall, precision)
        print(f"\n✅ GOLD PREDICTOR RESULTS:")
        print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}, ROC AUC: {roc_auc:.3f}, PR AUC: {pr_auc:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=3))

        # --- Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.title("LightGBM Confusion Matrix — Gold Predictor v3")
        plt.show()

        # --- Feature Importance ---
        import lightgbm as lgb
        lgb.plot_importance(model.model, max_num_features=15, importance_type="gain", figsize=(8, 5))
        plt.title("Top 15 Feature Importances — LightGBM v3")
        plt.show()


if __name__ == "__main__":
    Trainer().run()

# model/model.py
import lightgbm as lgb

class GoldPredictor:
    """LightGBM-based binary classifier for Gold direction prediction."""

    def __init__(self, class_ratio=1.0, random_state=42):
        self.model = lgb.LGBMClassifier(
            num_leaves=63,
            learning_rate=0.02,
            n_estimators=800,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_samples=25,
            objective="binary",
            scale_pos_weight=class_ratio,
            random_state=random_state,
            n_jobs=-1
        )

    def train(self, X_train, y_train, X_val, y_val):
        print("🚀 Training LightGBM model...")
        self.model.fit(
            X_train.clip(-5, 5), y_train,
            eval_set=[(X_val.clip(-5, 5), y_val)],
            eval_metric="logloss",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        print("✅ Training complete.")

    def predict(self, X):
        return self.model.predict(X.clip(-5, 5))

    def predict_proba(self, X):
        return self.model.predict_proba(X.clip(-5, 5))[:, 1]

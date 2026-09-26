"""Reproducible classifiers and held-out sigmoid calibration."""
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)


class ExactRuleModel:
    def fit(self, features, labels):
        return self

    def predict_proba(self, features):
        positive = ((features.name_normalized_exact == 1)
                    & ((features.address_normalized_exact == 1)
                       | ((features.postal_code_match == 1) & (features.street_number_match == 1)))
                    & ((features.country_match == 1) | (features.country_both_present == 0))).astype(float).to_numpy()
        return np.column_stack([1 - positive, positive])


def model_candidates(config, labels):
    try:
        from xgboost import XGBClassifier
    except (ImportError, OSError) as exc:
        raise RuntimeError("XGBoost is required. Install requirements.txt; on macOS its runtime may also need libomp.") from exc
    positives = max(1, int(np.sum(labels)))
    negatives = max(1, len(labels) - positives)
    ratio = min(15.0, negatives / positives)
    return {
        "exact_rule": ExactRuleModel(),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=config.seed)),
        "random_forest": RandomForestClassifier(n_estimators=config.trees, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample", n_jobs=config.threads, random_state=config.seed),
        "xgboost": XGBClassifier(n_estimators=config.trees, max_depth=6, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=4.0, min_child_weight=2,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            scale_pos_weight=ratio, n_jobs=config.threads, random_state=config.seed),
    }


class CalibratedMatcher:
    def __init__(self, estimator):
        self.estimator = estimator
        self.calibrator = None
        self.calibration_status = "not_requested"

    @staticmethod
    def logits(probabilities):
        p = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def calibrate(self, features, labels):
        if len(labels) == 0 or len(np.unique(labels)) < 2:
            self.calibration_status = "skipped_one_class_or_empty"
            LOG.warning("Calibration split lacks both labels; retaining raw probabilities")
            return self
        # No class weights here: calibration should reflect the actual candidate distribution.
        self.calibrator = LogisticRegression(C=1.0, max_iter=1000)
        self.calibrator.fit(self.logits(self.estimator.predict_proba(features)[:, 1]), labels)
        self.calibration_status = "sigmoid_on_disjoint_source1_split"
        return self

    def predict(self, features):
        if len(features) == 0:
            return np.empty(0, dtype=float)
        probabilities = self.estimator.predict_proba(features)[:, 1]
        if self.calibrator is not None:
            probabilities = self.calibrator.predict_proba(self.logits(probabilities))[:, 1]
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Model returned invalid probabilities")
        return probabilities

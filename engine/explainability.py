# engine/explainability.py
"""
SHAP Explainability Engine
---------------------------
Explains WHY each stop was assigned to each vehicle and sequenced in a
particular order. Uses SHAP TreeExplainer on a RandomForest that learns
to replicate the optimizer's stop-assignment decisions.

Outputs per stop:
  • Feature importance: which factors (weight, priority, deadline, distance)
    most influenced the assignment
  • SHAP waterfall: contribution of each factor to the final decision
  • Global summary: which features drive routing across all days

Usage:
    from engine.explainability import RouteExplainer
    explainer = RouteExplainer()
    explainer.fit(data, result)
    explanations = explainer.explain_all()
    report       = explainer.summary_report()
"""

import numpy as np
import json
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── SHAP ──────────────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not installed. Run: pip install shap")

# ── Sklearn ───────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class RouteExplainer:
    """
    Trains a surrogate model (RandomForest) on the optimizer's decisions,
    then applies SHAP to explain individual and global feature contributions.

    Features per stop:
      - weight_kg           : delivery weight
      - service_time        : time to unload at stop
      - earliest_arrival    : earliest the vehicle can arrive (EAT)
      - latest_arrival      : deadline (LAT)
      - time_window         : latest_arrival - earliest_arrival
      - priority_score      : URGENT=4, HIGH=3, MEDIUM=2, LOW=1
      - dist_from_depot     : distance matrix value [0][stop_id]
      - avg_dist_to_others  : mean distance to all other stops in same cluster

    Target: vehicle_id assigned by optimizer
    """

    PRIORITY_SCORE = {"URGENT": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    FEATURE_NAMES  = [
        "weight_kg", "service_time", "earliest_arrival", "latest_arrival",
        "time_window", "priority_score", "dist_from_depot", "avg_dist_to_others"
    ]

    def __init__(self):
        self._model:       Optional[RandomForestClassifier] = None
        self._explainer:   Optional["shap.TreeExplainer"]   = None
        self._shap_values: Optional[np.ndarray]              = None
        self._X:           Optional[pd.DataFrame]            = None
        self._y:           Optional[np.ndarray]              = None
        self._le:          LabelEncoder                      = LabelEncoder()
        self._data:        Optional[Dict]                     = None
        self._result:      Optional[Dict]                    = None
        self._stop_lookup: Dict                               = {}

    # ── Feature engineering ───────────────────────────────────────────────────
    def _build_features(self, data: Dict, result: Dict) -> pd.DataFrame:
        """
        Builds a feature matrix where each row = one stop,
        and the target = vehicle_id that served it.
        """
        stops      = data["stops"]
        dist_matrix = np.array(data["dist_matrix"])
        self._stop_lookup = {s["id"]: s for s in stops}

        # Build assignment map: stop_id → vehicle_id
        assignment = {}
        for route in result["optimized_routes"]:
            for node in route["stop_sequence"]:
                assignment[node] = route["vehicle_id"]

        rows   = []
        labels = []

        for stop in stops:
            sid = stop["id"]
            if sid not in assignment:
                continue   # unserved stop — skip for supervised learning

            dist_from_depot = dist_matrix[0][sid] if sid < len(dist_matrix) else 0.0
            # Average distance to all other stops
            all_stop_ids    = [s["id"] for s in stops if s["id"] != sid]
            avg_dist        = (np.mean([dist_matrix[sid][o]
                                        for o in all_stop_ids
                                        if o < len(dist_matrix)])
                               if all_stop_ids else 0.0)

            rows.append({
                "weight_kg":        stop["weight_kg"],
                "service_time":     stop["service_time"],
                "earliest_arrival": stop["earliest_arrival"],
                "latest_arrival":   stop["latest_arrival"],
                "time_window":      stop["latest_arrival"] - stop["earliest_arrival"],
                "priority_score":   self.PRIORITY_SCORE.get(stop["priority"], 1),
                "dist_from_depot":  dist_from_depot,
                "avg_dist_to_others": avg_dist
            })
            labels.append(assignment[sid])

        return pd.DataFrame(rows, columns=self.FEATURE_NAMES), np.array(labels)

    # ── Public API ────────────────────────────────────────────────────────────
    def fit(self, data: Dict, result: Dict):
        """
        Fits surrogate model + SHAP explainer on one day's data.

        Parameters
        ----------
        data   : preprocessed day JSON dict
        result : optimizer result JSON dict
        """
        self._data   = data
        self._result = result

        X, y = self._build_features(data, result)
        if len(X) == 0:
            print("⚠️  No assigned stops found — cannot fit explainer")
            return

        self._X = X
        self._y = self._le.fit_transform(y)

        # Use GradientBoosting if only 1 vehicle (binary), else RF
        n_classes = len(np.unique(self._y))
        if n_classes < 2:
            print("⚠️  Only 1 vehicle class — explainer needs ≥2 vehicles")
            return

        self._model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        self._model.fit(X, self._y)
        acc = self._model.score(X, self._y)
        print(f"✅ Surrogate model fitted | accuracy: {acc:.1%} "
              f"| {n_classes} vehicle classes")

        if SHAP_AVAILABLE:
            self._explainer   = shap.TreeExplainer(self._model)
            self._shap_values = self._explainer.shap_values(X)
            print(f"✅ SHAP explainer ready | {len(X)} stops explained")
        else:
            print("⚠️  SHAP not available — using feature_importances_ fallback")

    def explain_stop(self, stop_id: int) -> Dict:
        """
        Returns SHAP explanation for a single stop's vehicle assignment.

        Returns
        -------
        dict with keys: stop_id, assigned_vehicle, feature_contributions,
                        top_reason, confidence
        """
        if self._X is None or self._model is None:
            return {"error": "Explainer not fitted"}

        stops = self._data["stops"]
        idx   = next((i for i, s in enumerate(stops) if s["id"] == stop_id), None)
        if idx is None:
            return {"error": f"Stop {stop_id} not found"}

        # Which row in X corresponds to this stop?
        assignment = {}
        for route in self._result["optimized_routes"]:
            for node in route["stop_sequence"]:
                assignment[node] = route["vehicle_id"]

        if stop_id not in assignment:
            return {"error": f"Stop {stop_id} was not served"}

        # Find row index in X for this stop_id
        served_ids = list(assignment.keys())
        if stop_id not in served_ids:
            return {"error": "Stop not in feature matrix"}
        row_idx = served_ids.index(stop_id)

        x_row    = self._X.iloc[row_idx]
        vehicle  = assignment[stop_id]
        pred_cls = self._le.transform([vehicle])[0]
        proba    = self._model.predict_proba(x_row.values.reshape(1, -1))[0]
        confidence = float(proba[pred_cls])

        # Feature contributions
        if SHAP_AVAILABLE and self._shap_values is not None:
            # shap_values shape: (n_classes, n_samples, n_features) for RF
            sv = (self._shap_values[pred_cls][row_idx]
                  if isinstance(self._shap_values, list)
                  else self._shap_values[row_idx])
            contributions = {
                feat: round(float(sv[i]), 4)
                for i, feat in enumerate(self.FEATURE_NAMES)
            }
        else:
            # Fallback: use global feature importances
            importances = self._model.feature_importances_
            contributions = {
                feat: round(float(importances[i] * x_row.iloc[i]), 4)
                for i, feat in enumerate(self.FEATURE_NAMES)
            }

        # Top reason = feature with largest absolute SHAP value
        top_feature = max(contributions, key=lambda k: abs(contributions[k]))
        top_reason  = self._humanize(top_feature, x_row[top_feature],
                                      contributions[top_feature])

        return {
            "stop_id":              stop_id,
            "assigned_vehicle":     vehicle,
            "confidence":           round(confidence, 3),
            "feature_contributions": contributions,
            "top_reason":           top_reason,
            "feature_values":       x_row.to_dict()
        }

    def explain_all(self) -> List[Dict]:
        """Returns explanations for every served stop."""
        if self._data is None:
            return []
        assignment = {}
        for route in self._result["optimized_routes"]:
            for node in route["stop_sequence"]:
                assignment[node] = route["vehicle_id"]
        return [self.explain_stop(sid) for sid in assignment]

    def global_importance(self) -> Dict:
        """
        Returns mean absolute SHAP values per feature across all stops.
        Represents which factors most drive vehicle assignment globally.
        """
        if self._model is None:
            return {}

        if SHAP_AVAILABLE and self._shap_values is not None:
            if isinstance(self._shap_values, list):
                # Average across all classes
                mean_abs = np.mean([
                    np.abs(sv).mean(axis=0) for sv in self._shap_values
                ], axis=0)
            else:
                mean_abs = np.abs(self._shap_values).mean(axis=0)
        else:
            mean_abs = self._model.feature_importances_

        return {
            feat: round(float(mean_abs[i]), 4)
            for i, feat in enumerate(self.FEATURE_NAMES)
        }

    def summary_report(self) -> Dict:
        """Full explainability report: global importance + per-vehicle breakdown."""
        if self._model is None:
            return {"error": "Explainer not fitted"}

        global_imp = self.global_importance()
        ranked     = sorted(global_imp.items(), key=lambda x: -abs(x[1]))

        vehicle_breakdowns = {}
        for route in self._result["optimized_routes"]:
            vid   = route["vehicle_id"]
            expls = [self.explain_stop(n) for n in route["stop_sequence"]]
            valid = [e for e in expls if "error" not in e]
            if not valid:
                continue
            avg_conf  = np.mean([e["confidence"] for e in valid])
            top_feats = {}
            for e in valid:
                top = max(e["feature_contributions"],
                          key=lambda k: abs(e["feature_contributions"][k]))
                top_feats[top] = top_feats.get(top, 0) + 1
            vehicle_breakdowns[f"vehicle_{vid}"] = {
                "num_stops":            len(valid),
                "avg_confidence":       round(float(avg_conf), 3),
                "dominant_factor":      max(top_feats, key=top_feats.get)
                                        if top_feats else "N/A"
            }

        return {
            "global_feature_ranking": [
                {"feature": f, "importance": v, "rank": i + 1}
                for i, (f, v) in enumerate(ranked)
            ],
            "top_driver":           ranked[0][0] if ranked else "N/A",
            "vehicle_breakdowns":   vehicle_breakdowns,
            "shap_available":       SHAP_AVAILABLE,
            "model_accuracy":       round(
                float(self._model.score(self._X, self._y)), 3
            ) if self._X is not None else None
        }

    def _humanize(self, feature: str, value: float, shap_val: float) -> str:
        """Converts a SHAP feature + value into a human-readable sentence."""
        direction = "pushed toward" if shap_val > 0 else "pushed away from"
        messages  = {
            "weight_kg":          f"Heavy load ({value:.1f} kg) {direction} this vehicle",
            "service_time":       f"Long service time ({value:.0f} min) {direction} this vehicle",
            "earliest_arrival":   f"Early time window start ({value:.0f} min) {direction} this vehicle",
            "latest_arrival":     f"Tight deadline ({value:.0f} min) {direction} this vehicle",
            "time_window":        f"Narrow time window ({value:.0f} min) {direction} this vehicle",
            "priority_score":     f"High priority (score {value:.0f}) {direction} this vehicle",
            "dist_from_depot":    f"Distance from depot ({value:.1f} km) {direction} this vehicle",
            "avg_dist_to_others": f"Cluster isolation ({value:.1f} km avg) {direction} this vehicle"
        }
        return messages.get(feature, f"{feature}={value:.2f} {direction} this vehicle")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("data/processed/day1_mostlikely.json") as f:
        data = json.load(f)
    with open("data/results/day1_result.json") as f:
        result = json.load(f)

    explainer = RouteExplainer()
    explainer.fit(data, result)

    report = explainer.summary_report()
    print("\n📊 GLOBAL FEATURE IMPORTANCE (SHAP):")
    for item in report["global_feature_ranking"]:
        bar = "█" * int(abs(item["importance"]) * 200)
        print(f"  {item['rank']}. {item['feature']:25s} {item['importance']:+.4f}  {bar}")

    print(f"\n🏆 Top driver of vehicle assignment: {report['top_driver']}")
    print(f"🎯 Surrogate model accuracy: {report['model_accuracy']:.1%}")

    print("\n🔍 Sample stop explanation:")
    first_stop = result["optimized_routes"][0]["stop_sequence"][0]
    expl = explainer.explain_stop(first_stop)
    print(f"  Stop {first_stop} → Vehicle {expl['assigned_vehicle']} "
          f"(confidence: {expl['confidence']:.1%})")
    print(f"  Top reason: {expl['top_reason']}")
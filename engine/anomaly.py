# engine/anomaly.py
"""
Anomaly Detection Engine
-------------------------
Flags unusual patterns in route execution:
  • Route deviation  — actual travel time >> expected
  • Idle time spike  — vehicle stopped too long at a stop
  • Fuel spike       — distance travelled >> expected for this route
  • SLA breach       — arrival after latest_arrival deadline
  • Capacity overload— vehicle assigned more weight than capacity

Uses Isolation Forest (unsupervised) + rule-based thresholds.

Usage:
    from engine.anomaly import AnomalyDetector
    detector = AnomalyDetector()
    detector.fit(historical_results)         # learn normal patterns
    anomalies = detector.detect(result, data) # flag today's anomalies
"""

import numpy as np
import json
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════════════
#  Anomaly Detector
# ══════════════════════════════════════════════════════════════════════════════
class AnomalyDetector:
    """
    Two-layer anomaly detection:
      Layer 1 — Rule-based: immediate flags for clear violations
                (SLA breach, capacity overload, impossible time)
      Layer 2 — Isolation Forest: statistical outliers vs historical baseline
    """

    # Thresholds for rule-based detection
    IDLE_TIME_THRESHOLD_MIN   = 45    # >45 min at one stop = idle anomaly
    TRAVEL_MULTIPLIER         = 2.0   # actual > 2x expected = deviation
    FUEL_SPIKE_MULTIPLIER     = 1.8   # distance > 1.8x expected = fuel spike
    CAPACITY_BUFFER_KG        = 0     # no buffer — any overload is anomaly
    ISOLATION_CONTAMINATION   = 0.1   # expect 10% anomaly rate

    def __init__(self):
        self._iso_forest:  Optional[IsolationForest] = None
        self._scaler:      StandardScaler            = StandardScaler()
        self._fitted:      bool                      = False
        self._history_features: List[List[float]]   = []

    # ── Feature extraction per route ──────────────────────────────────────────
    def _route_features(
        self,
        route: Dict,
        data: Dict,
        vehicle_capacity_kg: float = 300.0
    ) -> Dict:
        """
        Extracts numerical features from one vehicle's route for anomaly scoring.
        """
        stop_lookup = {s["id"]: s for s in data["stops"]}
        nodes       = route["stop_sequence"]

        if not nodes:
            return {}

        # Expected travel time (from time matrix, no traffic)
        expected_time = 0.0
        actual_dist   = 0.0
        total_weight  = 0.0
        sla_violations = 0
        max_service_time = 0
        cumulative_time  = 0.0

        prev = 0  # depot
        for node in nodes:
            travel   = data["time_matrix"][prev][node]
            expected_time += travel
            actual_dist   += data["dist_matrix"][prev][node]

            if node in stop_lookup:
                s = stop_lookup[node]
                total_weight     += s["weight_kg"]
                service           = s["service_time"]
                expected_time    += service
                max_service_time  = max(max_service_time, service)
                cumulative_time  += travel + service
                # SLA check
                if cumulative_time > s["latest_arrival"]:
                    sla_violations += 1
            prev = node

        # Return to depot
        if nodes:
            actual_dist += data["dist_matrix"][nodes[-1]][0]

        actual_time  = route.get("travel_time_min", expected_time)
        actual_dist_r = route.get("distance_km", actual_dist)

        return {
            "vehicle_id":         route["vehicle_id"],
            "num_stops":          route["num_stops"],
            "expected_time_min":  round(expected_time, 1),
            "actual_time_min":    round(actual_time, 1),
            "time_ratio":         round(actual_time / max(expected_time, 1), 3),
            "expected_dist_km":   round(actual_dist, 2),
            "actual_dist_km":     round(actual_dist_r, 2),
            "dist_ratio":         round(actual_dist_r / max(actual_dist, 0.1), 3),
            "total_weight_kg":    round(total_weight, 2),
            "capacity_used_pct":  round(total_weight / vehicle_capacity_kg * 100, 1),
            "sla_violations":     sla_violations,
            "max_service_time":   max_service_time
        }

    def _feature_vector(self, feat: Dict) -> List[float]:
        """Converts feature dict to numeric vector for Isolation Forest."""
        return [
            feat.get("time_ratio", 1.0),
            feat.get("dist_ratio", 1.0),
            feat.get("capacity_used_pct", 0.0) / 100.0,
            feat.get("sla_violations", 0),
            feat.get("max_service_time", 0) / 60.0,
            feat.get("num_stops", 0) / 30.0
        ]

    # ── Rule-based detection ──────────────────────────────────────────────────
    def _rule_based_flags(self, feat: Dict, vehicle_capacity_kg: float) -> List[Dict]:
        """Returns list of rule-triggered anomaly dicts."""
        flags = []

        # Route deviation
        if feat["time_ratio"] > self.TRAVEL_MULTIPLIER:
            flags.append({
                "type":     "ROUTE_DEVIATION",
                "severity": "HIGH" if feat["time_ratio"] > 3.0 else "MEDIUM",
                "message":  (f"Travel time {feat['actual_time_min']} min is "
                             f"{feat['time_ratio']:.1f}x expected "
                             f"({feat['expected_time_min']} min)"),
                "value":    feat["time_ratio"]
            })

        # Fuel / distance spike
        if feat["dist_ratio"] > self.FUEL_SPIKE_MULTIPLIER:
            flags.append({
                "type":     "FUEL_SPIKE",
                "severity": "MEDIUM",
                "message":  (f"Distance {feat['actual_dist_km']} km is "
                             f"{feat['dist_ratio']:.1f}x expected "
                             f"({feat['expected_dist_km']} km)"),
                "value":    feat["dist_ratio"]
            })

        # SLA breach
        if feat["sla_violations"] > 0:
            flags.append({
                "type":     "SLA_BREACH",
                "severity": "HIGH",
                "message":  (f"{feat['sla_violations']} stop(s) will miss deadline "
                             f"based on planned sequence"),
                "value":    feat["sla_violations"]
            })

        # Capacity overload
        if feat["total_weight_kg"] > vehicle_capacity_kg:
            flags.append({
                "type":     "CAPACITY_OVERLOAD",
                "severity": "CRITICAL",
                "message":  (f"Vehicle loaded {feat['total_weight_kg']} kg — "
                             f"exceeds {vehicle_capacity_kg} kg capacity"),
                "value":    feat["total_weight_kg"]
            })

        # Idle time
        if feat["max_service_time"] > self.IDLE_TIME_THRESHOLD_MIN:
            flags.append({
                "type":     "IDLE_TIME_SPIKE",
                "severity": "LOW",
                "message":  (f"Stop requires {feat['max_service_time']} min service — "
                             f"possible idle or access issue"),
                "value":    feat["max_service_time"]
            })

        return flags

    # ── Public API ────────────────────────────────────────────────────────────
    def fit(self, historical_results: List[Dict], historical_data: List[Dict]):
        """
        Fits Isolation Forest on historical route features to learn
        what 'normal' looks like.

        Parameters
        ----------
        historical_results : list of result dicts (one per past day)
        historical_data    : list of data dicts (matching days)
        """
        all_features = []
        for result, data in zip(historical_results, historical_data):
            for route in result.get("optimized_routes", []):
                feat = self._route_features(route, data)
                if feat:
                    all_features.append(self._feature_vector(feat))

        if len(all_features) < 5:
            print(f"⚠️  Only {len(all_features)} historical routes — "
                  f"Isolation Forest needs more data. Using rules only.")
            return

        self._history_features = all_features
        X = self._scaler.fit_transform(all_features)
        self._iso_forest = IsolationForest(
            contamination=self.ISOLATION_CONTAMINATION,
            random_state=42,
            n_estimators=100
        )
        self._iso_forest.fit(X)
        self._fitted = True
        print(f"✅ Anomaly detector fitted on {len(all_features)} historical routes")

    def detect(
        self,
        result: Dict,
        data: Dict,
        vehicle_capacity_kg: float = 300.0
    ) -> List[Dict]:
        """
        Detects anomalies in today's routes.

        Returns
        -------
        List of anomaly records:
          {vehicle_id, anomaly_type, severity, message, value,
           detection_method, features}
        """
        anomalies = []

        for route in result.get("optimized_routes", []):
            feat  = self._route_features(route, data, vehicle_capacity_kg)
            if not feat:
                continue

            vid   = route["vehicle_id"]

            # Layer 1: rule-based
            flags = self._rule_based_flags(feat, vehicle_capacity_kg)
            for flag in flags:
                anomalies.append({
                    "vehicle_id":       vid,
                    "detection_method": "rule-based",
                    "features":         feat,
                    **flag
                })

            # Layer 2: Isolation Forest
            if self._fitted and self._iso_forest is not None:
                fv = np.array(self._feature_vector(feat)).reshape(1, -1)
                fv_scaled = self._scaler.transform(fv)
                score     = self._iso_forest.score_samples(fv_scaled)[0]
                is_anomaly = self._iso_forest.predict(fv_scaled)[0] == -1

                if is_anomaly:
                    anomalies.append({
                        "vehicle_id":        vid,
                        "type":              "STATISTICAL_OUTLIER",
                        "severity":          "MEDIUM",
                        "message":           (f"Vehicle {vid}'s route pattern is "
                                              f"statistically unusual "
                                              f"(anomaly score: {score:.3f})"),
                        "value":             round(score, 4),
                        "detection_method":  "isolation_forest",
                        "features":          feat
                    })

        return anomalies

    def anomaly_summary(self, anomalies: List[Dict]) -> Dict:
        """Aggregates anomalies into a dashboard-ready summary."""
        if not anomalies:
            return {
                "total": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "by_type": {},
                "vehicles_affected": []
            }

        from collections import Counter
        severities = Counter(a["severity"] for a in anomalies)
        by_type    = Counter(a["type"]     for a in anomalies)
        vehicles   = list(set(a["vehicle_id"] for a in anomalies))

        return {
            "total":              len(anomalies),
            "critical":           severities.get("CRITICAL", 0),
            "high":               severities.get("HIGH", 0),
            "medium":             severities.get("MEDIUM", 0),
            "low":                severities.get("LOW", 0),
            "by_type":            dict(by_type),
            "vehicles_affected":  sorted(vehicles),
            "recommendations":    self._recommendations(anomalies)
        }

    def _recommendations(self, anomalies: List[Dict]) -> List[str]:
        """Generates human-readable action recommendations."""
        recs = []
        types = {a["type"] for a in anomalies}

        if "CAPACITY_OVERLOAD" in types:
            recs.append("🚨 Reassign stops from overloaded vehicles before dispatch")
        if "SLA_BREACH" in types:
            affected = [a["vehicle_id"] for a in anomalies if a["type"] == "SLA_BREACH"]
            recs.append(f"⏰ Vehicles {affected} have SLA-risk stops — "
                        f"consider re-sequencing or adding a vehicle")
        if "ROUTE_DEVIATION" in types:
            recs.append("🗺 Check for traffic incidents or GPS issues on flagged routes")
        if "FUEL_SPIKE" in types:
            recs.append("⛽ Verify distance matrix accuracy or check for detours")
        if "IDLE_TIME_SPIKE" in types:
            recs.append("⏸ Investigate access issues at stops with long service times")
        if "STATISTICAL_OUTLIER" in types:
            recs.append("📊 Unusual route patterns detected — manual review recommended")

        return recs


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import glob

    result_paths = sorted(glob.glob("data/results/day*_result.json"))
    data_paths   = sorted(glob.glob("data/processed/day*_mostlikely.json"))

    if not result_paths:
        print("❌ No results found. Run optimizer first.")
    else:
        historical_results = []
        historical_data    = []
        for rp, dp in zip(result_paths[:-1], data_paths[:-1]):
            with open(rp) as f: historical_results.append(json.load(f))
            with open(dp) as f: historical_data.append(json.load(f))

        detector = AnomalyDetector()
        detector.fit(historical_results, historical_data)

        # Detect on latest day
        with open(result_paths[-1]) as f:  result = json.load(f)
        with open(data_paths[-1])   as f:  data   = json.load(f)

        anomalies = detector.detect(result, data)
        summary   = detector.anomaly_summary(anomalies)

        print(f"\n🚨 ANOMALY REPORT — Day {result['day']}")
        print(f"   Total anomalies  : {summary['total']}")
        print(f"   Critical         : {summary['critical']}")
        print(f"   High             : {summary['high']}")
        print(f"   Medium           : {summary['medium']}")
        print(f"   Vehicles affected: {summary['vehicles_affected']}")

        if anomalies:
            print("\n📋 DETAILED ANOMALIES:")
            for a in anomalies:
                icon = {"CRITICAL": "🔴", "HIGH": "🟠",
                        "MEDIUM": "🟡", "LOW": "🔵"}.get(a["severity"], "⚪")
                print(f"  {icon} Vehicle {a['vehicle_id']} | "
                      f"{a['type']:25s} | {a['message']}")

        if summary["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for r in summary["recommendations"]:
                print(f"  {r}")
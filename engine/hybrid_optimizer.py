# engine/hybrid_optimizer.py
"""
Hybrid Optimizer — OR-Tools ↔ RL Decision Layer
-------------------------------------------------
Orchestrates the full pipeline:
  1. Clustering  → split stops into vehicle zones
  2. OR-Tools    → compute optimal static routes per cluster
  3. RL Agent    → optionally re-route based on real-time traffic
  4. SHAP        → explain final route decisions
  5. Anomaly     → flag any issues in the final plan

This is the single entry point called by optimizer.py and dashboard.py.

Usage:
    from engine.hybrid_optimizer import HybridOptimizer
    opt = HybridOptimizer(num_vehicles=5)
    result = opt.run(data, traffic_multiplier=1.0, use_rl=True)
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

# ── Internal engines ──────────────────────────────────────────────────────────
from engine.clustering     import ClusteringEngine
from engine.explainability import RouteExplainer
from engine.anomaly        import AnomalyDetector

# ── RL agent (graceful optional) ─────────────────────────────────────────────
try:
    from engine.rl_agent import RLReRouter
    RL_AVAILABLE = True
except Exception:
    RL_AVAILABLE = False

# ── OR-Tools ──────────────────────────────────────────────────────────────────
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

PRIORITY_PENALTY = {
    "URGENT": 1_000_000,
    "HIGH":   100_000,
    "MEDIUM": 50_000,
    "LOW":    10_000
}


def _build_stop_lookup(stops):
    return {s["id"]: s for s in stops}


def _compute_route_metrics(route_node_ids, data):
    stop_lookup = _build_stop_lookup(data["stops"])
    total_time = 0.0
    total_dist = 0.0
    route = [0] + route_node_ids + [0]
    for i in range(len(route) - 1):
        frm, to = route[i], route[i + 1]
        total_time += data["time_matrix"][frm][to]
        total_dist += data["dist_matrix"][frm][to]
        if to != 0:
            total_time += stop_lookup[to]["service_time"]
    return round(total_time, 1), round(total_dist, 2)


def _build_naive_route(data):
    sorted_stops = sorted(data["stops"], key=lambda s: s["latest_arrival"])
    return [s["id"] for s in sorted_stops]


def _build_nn_route(data):
    """
    Nearest-neighbor greedy heuristic: start at depot, always go to
    the closest unvisited stop. A fairer baseline than naive sequential.
    """
    stops     = data["stops"]
    tm        = data["time_matrix"]
    unvisited = {s["id"] for s in stops}
    route     = []
    current   = 0  # depot

    while unvisited:
        nearest = min(unvisited, key=lambda n: tm[current][n])
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return route


def _run_ortools(data, num_vehicles, vehicle_capacity_kg, solve_seconds=10):
    """
    Runs OR-Tools CVRPTW on a (possibly clustered sub-) data dict.
    Returns list of route dicts.
    """
    num_nodes   = len(data["time_matrix"])
    depot       = 0
    stop_lookup = _build_stop_lookup(data["stops"])

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        travel  = data["time_matrix"][i][j]
        service = stop_lookup[i]["service_time"] if i != depot else 0
        return travel + service

    tc_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(tc_idx)

    routing.AddDimension(tc_idx, 60, 480, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for stop in data["stops"]:
        node_idx = manager.NodeToIndex(stop["id"])
        time_dim.CumulVar(node_idx).SetRange(
            stop["earliest_arrival"], stop["latest_arrival"]
        )
        routing.AddDisjunction([node_idx], PRIORITY_PENALTY[stop["priority"]])

    def weight_callback(from_idx):
        node = manager.IndexToNode(from_idx)
        return int(stop_lookup[node]["weight_kg"] * 10) if node != depot else 0

    w_idx = routing.RegisterUnaryTransitCallback(weight_callback)
    routing.AddDimensionWithVehicleCapacity(
        w_idx, 0,
        [int(vehicle_capacity_kg * 10)] * num_vehicles,
        True, "Weight"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = solve_seconds

    solution = routing.SolveWithParameters(params)
    if not solution:
        return []

    routes = []
    for v in range(num_vehicles):
        nodes = []
        idx   = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != depot:
                nodes.append(node)
            idx = solution.Value(routing.NextVar(idx))
        if nodes:
            t, d = _compute_route_metrics(nodes, data)
            routes.append({
                "vehicle_id":      v + 1,
                "stop_sequence":   nodes,
                "num_stops":       len(nodes),
                "travel_time_min": t,
                "distance_km":     d
            })
    return routes


# ══════════════════════════════════════════════════════════════════════════════
#  HybridOptimizer
# ══════════════════════════════════════════════════════════════════════════════
class HybridOptimizer:
    """
    Full pipeline orchestrator:
      Clustering → OR-Tools → (optional RL) → SHAP → Anomaly Detection
    """

    def __init__(
        self,
        num_vehicles:        int   = 5,
        vehicle_capacity_kg: float = 300.0,
        use_clustering:      bool  = True,
        rl_model_path:       Optional[str] = None,
        solve_seconds:       int   = 10
    ):
        self.num_vehicles        = num_vehicles
        self.vehicle_capacity    = vehicle_capacity_kg
        self.use_clustering      = use_clustering
        self.solve_seconds       = solve_seconds

        self.clusterer   = ClusteringEngine(n_clusters=num_vehicles)
        self.explainer   = RouteExplainer()
        self.anomaly_det = AnomalyDetector()

        self.rl_agent: Optional["RLReRouter"] = None
        if RL_AVAILABLE and rl_model_path:
            self.rl_agent = RLReRouter(model_path=rl_model_path)

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(
        self,
        data:               Dict,
        traffic_multiplier: float = 1.0,
        use_rl:             bool  = False,
        explain:            bool  = True,
        detect_anomalies:   bool  = True
    ) -> Dict:
        """
        Runs the full hybrid optimization pipeline.

        Parameters
        ----------
        data               : preprocessed day JSON dict
        traffic_multiplier : 1.0=normal, >1=congested (triggers RL if use_rl=True)
        use_rl             : whether to apply RL re-routing after OR-Tools
        explain            : whether to run SHAP explainability
        detect_anomalies   : whether to run anomaly detection

        Returns
        -------
        Enriched result dict with routes, metrics, explanations, anomalies
        """
        print(f"\n{'='*60}")
        print(f"🚀 HybridOptimizer — Day {data['day']} | {data['scenario']}")
        print(f"   Vehicles: {self.num_vehicles} | Traffic: {traffic_multiplier}x")
        print(f"   Pipeline: OR-Tools"
              + (" + RL" if use_rl and RL_AVAILABLE else "")
              + (" + SHAP" if explain else "")
              + (" + Anomaly" if detect_anomalies else ""))
        print(f"{'='*60}")

        # ── Step 1: Clustering ────────────────────────────────────────────────
        cluster_info = {}
        if self.use_clustering and len(data["stops"]) > self.num_vehicles:
            print(f"\n📍 Step 1: Geo-clustering ({len(data['stops'])} stops → "
                  f"{self.num_vehicles} zones)...")
            labels, _ = self.clusterer.fit(data["stops"], data["dist_matrix"])
            cluster_info = self.clusterer.cluster_summary(data["stops"], labels)
            sub_datasets = self.clusterer.split_by_cluster(data, labels)
            print(f"   ✅ {len(sub_datasets)} zones created | "
                  f"Outlier stops: {cluster_info.get('outlier_stop_ids', [])}")
        else:
            sub_datasets = [data]
            labels       = None

        # ── Step 2: OR-Tools on each cluster ─────────────────────────────────
        print(f"\n⚙️  Step 2: OR-Tools CVRPTW optimization...")
        all_routes   = []
        vehicle_offset = 0

        # Distribute vehicles proportionally across clusters by stop count
        total_stops_all = sum(sub["num_stops"] for sub in sub_datasets)
        n_clusters = len(sub_datasets)

        for sub in sub_datasets:
            if n_clusters > 1 and total_stops_all > 0:
                # Proportional share, minimum 1 per cluster
                share = sub["num_stops"] / total_stops_all
                n_v = max(1, round(self.num_vehicles * share))
            else:
                n_v = self.num_vehicles
            n_v = min(n_v, sub["num_stops"])  # can't have more vehicles than stops

            sub_routes = _run_ortools(
                sub, n_v, self.vehicle_capacity, self.solve_seconds
            )

            # Remap sub-problem node IDs back to original if clustered
            if "_id_map" in sub and sub_routes:
                reverse_map = {v: k for k, v in sub["_id_map"].items()}
                for r in sub_routes:
                    r["stop_sequence"] = [
                        reverse_map.get(n, n) for n in r["stop_sequence"]
                    ]
                    r["vehicle_id"] += vehicle_offset
                vehicle_offset += len(sub_routes)

            all_routes.extend(sub_routes)

        if not all_routes:
            print("❌ OR-Tools found no solution")
            return {}

        print(f"   ✅ {len(all_routes)} vehicle routes computed | "
              f"{sum(r['num_stops'] for r in all_routes)} stops served")

        # Build intermediate result
        makespan   = max(r["travel_time_min"] for r in all_routes)
        total_dist = sum(r["distance_km"]     for r in all_routes)
        naive_order = _build_naive_route(data)
        nn_order    = _build_nn_route(data)

        naive_time, naive_dist = _compute_route_metrics(naive_order, data)
        nn_time, nn_dist       = _compute_route_metrics(nn_order, data)

        result = {
            "day":                 data["day"],
            "scenario":            data["scenario"],
            "num_stops":           data["num_stops"],
            "num_vehicles_used":   len(all_routes),
            "optimized_routes":    all_routes,
            "total_opt_time_min":  makespan,
            "total_opt_dist_km":   round(total_dist, 2),
            "naive_time_min":      naive_time,
            "naive_dist_km":       naive_dist,
            "nn_time_min":         nn_time,
            "nn_dist_km":          nn_dist,
            "time_saved_min":      round(naive_time - makespan, 1),
            "dist_saved_km":       round(naive_dist - total_dist, 2),
            "nn_time_saved_min":   round(nn_time - makespan, 1),
            "nn_dist_saved_km":    round(nn_dist - total_dist, 2),
            "cost_saved_inr":      round((naive_dist - total_dist) * 2.5, 2),
            "efficiency_gain_pct": round(
                ((naive_time - makespan) / naive_time * 100), 1
            ) if naive_time > 0 else 0,
            "nn_efficiency_gain_pct": round(
                ((nn_time - makespan) / nn_time * 100), 1
            ) if nn_time > 0 else 0,
            "pipeline": {
                "clustering_used": self.use_clustering and labels is not None,
                "rl_applied":      False,
                "traffic_factor":  traffic_multiplier
            }
        }

        # ── Step 3: RL re-routing (if traffic is significant) ─────────────────
        if use_rl and RL_AVAILABLE and traffic_multiplier > 1.1:
            if self.rl_agent is None:
                print(f"\n🤖 Step 3: Training RL agent (traffic={traffic_multiplier}x)...")
                self.rl_agent = RLReRouter()
                self.rl_agent.train(data, total_timesteps=20_000, verbose=0)
            else:
                print(f"\n🤖 Step 3: Applying RL re-routing (traffic={traffic_multiplier}x)...")

            result        = self.rl_agent.reroute(data, result, traffic_multiplier)
            result["pipeline"]["rl_applied"] = True
            print(f"   ✅ RL re-routing applied to "
                  f"{sum(1 for r in result['optimized_routes'] if r.get('rerouted'))} vehicles")
        else:
            if use_rl:
                print(f"\n🤖 Step 3: RL skipped (traffic={traffic_multiplier}x ≤ 1.1 threshold)")

        # ── Step 4: SHAP Explainability ───────────────────────────────────────
        if explain:
            print(f"\n🔍 Step 4: SHAP explainability...")
            try:
                self.explainer.fit(data, result)
                shap_report = self.explainer.summary_report()
                result["explainability"] = shap_report
                print(f"   ✅ Top driver: {shap_report.get('top_driver', 'N/A')} | "
                      f"Accuracy: {shap_report.get('model_accuracy', 'N/A')}")
            except Exception as e:
                print(f"   ⚠️  SHAP skipped: {e}")
                result["explainability"] = {"error": str(e)}

        # ── Step 5: Anomaly Detection ─────────────────────────────────────────
        if detect_anomalies:
            print(f"\n🚨 Step 5: Anomaly detection...")
            anomalies = self.anomaly_det.detect(result, data, self.vehicle_capacity)
            summary   = self.anomaly_det.anomaly_summary(anomalies)
            result["anomalies"]         = anomalies
            result["anomaly_summary"]   = summary
            result["cluster_info"]      = cluster_info
            print(f"   ✅ {summary['total']} anomalies detected | "
                  f"Critical: {summary['critical']} | High: {summary['high']}")

        print(f"\n✅ Pipeline complete — "
              f"makespan: {result['total_opt_time_min']} min | "
              f"efficiency: {result['efficiency_gain_pct']}%\n")

        return result

    def train_rl(self, data: Dict, timesteps: int = 50_000) -> None:
        """Train / retrain the RL agent and save to ml_model/."""
        if not RL_AVAILABLE:
            print("⚠️  stable-baselines3 not available")
            return
        self.rl_agent = RLReRouter()
        self.rl_agent.train(data, total_timesteps=timesteps)
        os.makedirs("ml_model", exist_ok=True)
        self.rl_agent.save(f"ml_model/ppo_router_day{data['day']}")

    def fit_anomaly_detector(
        self,
        result_paths: List[str],
        data_paths:   List[str]
    ) -> None:
        """Pre-fit anomaly detector from historical results."""
        results, datas = [], []
        for rp, dp in zip(result_paths, data_paths):
            if os.path.exists(rp) and os.path.exists(dp):
                with open(rp) as f: results.append(json.load(f))
                with open(dp) as f: datas.append(json.load(f))
        if results:
            self.anomaly_det.fit(results, datas)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--day",      type=int,   default=1)
    parser.add_argument("--scenario", type=str,   default="mostlikely")
    parser.add_argument("--vehicles", type=int,   default=5)
    parser.add_argument("--traffic",  type=float, default=1.0)
    parser.add_argument("--rl",       action="store_true")
    args = parser.parse_args()

    with open(f"data/processed/day{args.day}_{args.scenario}.json") as f:
        data = json.load(f)

    optimizer = HybridOptimizer(
        num_vehicles=args.vehicles,
        use_clustering=True
    )
    result = optimizer.run(
        data,
        traffic_multiplier=args.traffic,
        use_rl=args.rl,
        explain=True,
        detect_anomalies=True
    )

    if result:
        os.makedirs("data/results", exist_ok=True)
        out = f"data/results/day{args.day}_result.json"
        # Remove non-serializable numpy types before saving
        import copy
        save_result = copy.deepcopy(result)
        with open(out, "w") as f:
            json.dump(save_result, f, indent=2, default=str)
        print(f"✅ Result saved to {out}")
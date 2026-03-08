# optimizer.py
"""
Main optimizer entry point.
Delegates to engine/hybrid_optimizer.py which runs the full pipeline:
  Clustering → OR-Tools → RL (optional) → SHAP → Anomaly Detection
"""
import json, os, argparse, copy, glob

from engine.hybrid_optimizer import HybridOptimizer
from typing import Dict


def optimize(
    data:                Dict,
    num_vehicles:        int   = 5,
    vehicle_capacity_kg: float = 300.0,
    traffic_multiplier:  float = 1.0,
    use_rl:              bool  = False,
    use_clustering:      bool  = True,
    rl_model_path:       str   = None
) -> Dict:
    """
    Public API — called by dashboard.py and hybrid_optimizer standalone.

    Parameters
    ----------
    data                : preprocessed day JSON dict
    num_vehicles        : fleet size
    vehicle_capacity_kg : max payload per vehicle (kg)
    traffic_multiplier  : 1.0=normal, 1.5=heavy — triggers RL if >1.1 + use_rl=True
    use_rl              : activate PPO re-routing layer
    use_clustering      : activate K-Means/DBSCAN zone pre-grouping
    rl_model_path       : path to pre-trained PPO model (optional)

    Returns
    -------
    Full enriched result dict
    """
    optimizer = HybridOptimizer(
        num_vehicles        = num_vehicles,
        vehicle_capacity_kg = vehicle_capacity_kg,
        use_clustering      = use_clustering,
        rl_model_path       = rl_model_path,
        solve_seconds       = 10
    )

    # Pre-fit anomaly detector from historical results if available

    result_paths = sorted(glob.glob("data/results/day*_result.json"))
    data_paths   = [p.replace("results/day", "processed/day")
                      .replace("_result.json", "_mostlikely.json")
                    for p in result_paths]
    if len(result_paths) >= 2:
        optimizer.fit_anomaly_detector(result_paths[:-1], data_paths[:-1])

    return optimizer.run(
        data,
        traffic_multiplier = traffic_multiplier,
        use_rl             = use_rl,
        explain            = True,
        detect_anomalies   = True
    )


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RouteAI Optimizer")
    parser.add_argument("--day",      type=int,   default=1)
    parser.add_argument("--scenario", type=str,   default="mostlikely")
    parser.add_argument("--vehicles", type=int,   default=5)
    parser.add_argument("--traffic",  type=float, default=1.0,
                        help="Traffic multiplier (1.0=normal, 1.5=heavy)")
    parser.add_argument("--rl",       action="store_true",
                        help="Enable PPO RL re-routing layer")
    parser.add_argument("--no-cluster", action="store_true",
                        help="Disable geo-clustering pre-step")
    args = parser.parse_args()

    path = f"data/processed/day{args.day}_{args.scenario}.json"
    if not os.path.exists(path):
        print(f"❌ Data not found: {path}. Run preprocess.py first.")
        exit(1)

    with open(path) as f:
        data = json.load(f)

    result = optimize(
        data,
        num_vehicles        = args.vehicles,
        traffic_multiplier  = args.traffic,
        use_rl              = args.rl,
        use_clustering      = not args.no_cluster
    )

    if result:
        os.makedirs("data/results", exist_ok=True)
        out = f"data/results/day{args.day}_result.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n✅ Full pipeline result saved to: {out}")

        # Print summary
        print(f"\n📊 SUMMARY")
        print(f"   Vehicles used    : {result['num_vehicles_used']}")
        print(f"   Stops served     : "
              f"{sum(r['num_stops'] for r in result['optimized_routes'])}"
              f"/{result['num_stops']}")
        print(f"   Makespan         : {result['total_opt_time_min']} min")
        print(f"   Naive time       : {result['naive_time_min']} min")
        print(f"   Time saved       : {result['time_saved_min']} min")
        print(f"   Efficiency gain  : {result['efficiency_gain_pct']}%")
        print(f"   Cost saved       : ₹{result['cost_saved_inr']}")

        if result.get("anomaly_summary"):
            s = result["anomaly_summary"]
            print(f"\n🚨 ANOMALIES: {s['total']} total | "
                  f"Critical: {s['critical']} | High: {s['high']}")

        if result.get("explainability", {}).get("top_driver"):
            print(f"\n🔍 TOP ROUTE DRIVER: "
                  f"{result['explainability']['top_driver']}")
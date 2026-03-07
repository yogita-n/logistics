# optimizer.py
import json
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# ── Priority penalty map (higher = must serve first) ──────────────────────────
PRIORITY_PENALTY = {
    "URGENT": 1_000_000,
    "HIGH":   100_000,
    "MEDIUM": 50_000,
    "LOW":    10_000
}

# ── Build a fast stop lookup by node ID ───────────────────────────────────────
def build_stop_lookup(data):
    """Returns a dict: {node_id -> stop_dict} for O(1) access."""
    return {s["id"]: s for s in data["stops"]}

def build_naive_route(data):
    """
    Naive route: visit ALL stops with a single vehicle, ordered by tightest
    deadline (smallest deadline first — i.e. smallest LAT/latest_arrival).
    This simulates what an unoptimized delivery agent would do.
    """
    sorted_stops = sorted(data["stops"], key=lambda s: s["latest_arrival"])
    return [s["id"] for s in sorted_stops]

def compute_route_metrics(route_node_ids, data):
    """
    Given a list of node IDs (0=depot, 1..N=stops),
    compute total travel time (min) and total distance (km).
    Includes service time at each stop.
    """
    stop_lookup = build_stop_lookup(data)
    total_time = 0
    total_dist = 0
    route = [0] + route_node_ids + [0]  # start and end at depot (node 0)
    for i in range(len(route) - 1):
        frm = route[i]
        to  = route[i + 1]
        total_time += data["time_matrix"][frm][to]
        total_dist += data["dist_matrix"][frm][to]
        # Add service time at the destination stop (not at depot)
        if to != 0:
            total_time += stop_lookup[to]["service_time"]
    return round(total_time, 1), round(total_dist, 2)

def optimize(data, num_vehicles=5, vehicle_capacity_kg=300, vehicle_capacity_vol=2.0):
    """
    Runs OR-Tools VRPTW on the loaded data.
    Returns optimized routes + fair comparison metrics vs naive single-vehicle routing.

    Metric comparison approach:
    - Naive   : 1 vehicle visits all stops sequentially → total elapsed time
    - Optimized: N vehicles in parallel → the LONGEST single vehicle route determines
                 when ALL deliveries are done (makespan). This is the correct comparison
                 because customers don't care which truck serves them — they care when
                 the last delivery is completed.
    """
    num_nodes = len(data["time_matrix"])   # depot + N stops
    depot     = 0
    stop_lookup = build_stop_lookup(data)

    # ── OR-Tools Setup ────────────────────────────────────────────────────────
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # ── Time Callback ─────────────────────────────────────────────────────────
    # Uses safe stop lookup instead of fragile index arithmetic
    def time_callback(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        travel = data["time_matrix"][i][j]
        # Add service time at the origin node (charged when leaving a stop)
        service = stop_lookup[i]["service_time"] if i != depot else 0
        return travel + service

    time_transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(time_transit_idx)

    # ── Time Dimension ────────────────────────────────────────────────────────
    routing.AddDimension(
        time_transit_idx,
        60,     # max waiting/slack time allowed at a stop (minutes)
        480,    # max total route duration per vehicle (8 hours = 480 min)
        False,  # do not force start cumul to zero
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # ── Apply Time Windows + Priority Penalties per stop ──────────────────────
    for stop in data["stops"]:
        node_idx = manager.NodeToIndex(stop["id"])
        # Time window: [earliest_arrival, latest_arrival]
        time_dim.CumulVar(node_idx).SetRange(
            stop["earliest_arrival"],
            stop["latest_arrival"]
        )
        # Priority penalty: cost paid if this stop is skipped
        penalty = PRIORITY_PENALTY[stop["priority"]]
        routing.AddDisjunction([node_idx], penalty)

    # ── Weight Capacity Constraint ────────────────────────────────────────────
    # Multiply by 10 to keep integer arithmetic (OR-Tools requires integers)
    def weight_callback(from_idx):
        node = manager.IndexToNode(from_idx)
        return int(stop_lookup[node]["weight_kg"] * 10) if node != depot else 0

    weight_idx = routing.RegisterUnaryTransitCallback(weight_callback)
    routing.AddDimensionWithVehicleCapacity(
        weight_idx,
        0,
        [int(vehicle_capacity_kg * 10)] * num_vehicles,
        True,
        "Weight"
    )

    # ── Solver Parameters ─────────────────────────────────────────────────────
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = 10

    # ── Solve ─────────────────────────────────────────────────────────────────
    solution = routing.SolveWithParameters(params)

    if not solution:
        print("❌ No solution found!")
        return None

    # ── Extract Optimized Routes ──────────────────────────────────────────────
    optimized_routes = []
    max_vehicle_time = 0   # FIX: makespan = longest single vehicle route
    total_opt_dist   = 0

    for vehicle in range(num_vehicles):
        route_nodes = []
        index = routing.Start(vehicle)

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != depot:
                route_nodes.append(node)
            index = solution.Value(routing.NextVar(index))

        if route_nodes:
            t, d = compute_route_metrics(route_nodes, data)
            # FIX: track the LONGEST vehicle route (makespan), not sum
            max_vehicle_time = max(max_vehicle_time, t)
            total_opt_dist  += d
            optimized_routes.append({
                "vehicle_id":      vehicle + 1,
                "stop_sequence":   route_nodes,
                "num_stops":       len(route_nodes),
                "travel_time_min": t,
                "distance_km":     d
            })

    # ── Naive Route Metrics (single vehicle, all stops, tightest-deadline order)
    naive_order            = build_naive_route(data)
    naive_time, naive_dist = compute_route_metrics(naive_order, data)

    # ── Guard: savings can't be negative in the reported metrics ──────────────
    # (Optimized dist may exceed naive if stops are spread across many vehicles
    #  with depot returns — this is expected and honest to show)
    time_saved = round(naive_time - max_vehicle_time, 1)
    dist_saved = round(naive_dist - total_opt_dist,   2)

    # FIX: efficiency gain = time reduction relative to naive single-vehicle time
    efficiency_gain = 0.0
    if naive_time > 0:
        efficiency_gain = round((time_saved / naive_time) * 100, 1)
    # Clamp to [-100, 100] — large negative values indicate the problem is harder
    # than naive (e.g. many depot returns), which is honest information
    efficiency_gain = max(-100.0, min(100.0, efficiency_gain))

    # ── Summary ───────────────────────────────────────────────────────────────
    result = {
        "day":                    data["day"],
        "scenario":               data["scenario"],
        "num_stops":              data["num_stops"],
        "num_vehicles_used":      len(optimized_routes),
        "optimized_routes":       optimized_routes,
        # FIX: use makespan (longest vehicle time), not sum of all vehicles
        "total_opt_time_min":     round(max_vehicle_time, 1),
        "total_opt_dist_km":      round(total_opt_dist, 2),
        "naive_time_min":         round(naive_time, 1),
        "naive_dist_km":          round(naive_dist, 2),
        "time_saved_min":         time_saved,
        "dist_saved_km":          dist_saved,
        # FIX: cost based on distance saved; show actual value (can be negative)
        "cost_saved_inr":         round(dist_saved * 2.5, 2),
        "efficiency_gain_pct":    efficiency_gain
    }
    return result


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--day",      type=int, default=1)
    parser.add_argument("--scenario", type=str, default="mostlikely")
    parser.add_argument("--vehicles", type=int, default=5)
    args = parser.parse_args()

    path = f"data/processed/day{args.day}_{args.scenario}.json"
    with open(path) as f:
        data = json.load(f)

    print(f"🚀 Running optimization for Day {args.day} ({args.scenario})...")
    result = optimize(data, num_vehicles=args.vehicles, vehicle_capacity_kg=300)

    if result:
        print(f"\n✅ OPTIMIZATION COMPLETE")
        print(f"   Vehicles used      : {result['num_vehicles_used']}")
        print(f"   Total stops served : {sum(r['num_stops'] for r in result['optimized_routes'])}")

        print(f"\n📊 PERFORMANCE COMPARISON (Naive=1 vehicle vs Optimized makespan)")
        print(f"   {'Metric':<28} {'Naive':>10} {'Optimized':>12} {'Saved':>10}")
        print(f"   {'-'*62}")
        print(f"   {'Total time (min)':<28} {result['naive_time_min']:>10} "
              f"{result['total_opt_time_min']:>12} {result['time_saved_min']:>10}")
        print(f"   {'Total distance (km)':<28} {result['naive_dist_km']:>10} "
              f"{result['total_opt_dist_km']:>12} {result['dist_saved_km']:>10}")
        print(f"   {'Cost saved (INR ₹)':<28} {'':>10} {'':>12} "
              f"{result['cost_saved_inr']:>10}")
        print(f"   {'Efficiency gain':<28} {'':>10} {'':>12} "
              f"{result['efficiency_gain_pct']:>9}%")

        print(f"\n🗺  OPTIMIZED ROUTES PER VEHICLE")
        for r in result["optimized_routes"]:
            print(f"   Vehicle {r['vehicle_id']}: {r['num_stops']} stops | "
                  f"{r['travel_time_min']} min | {r['distance_km']} km")
            print(f"     → Stops: {r['stop_sequence']}")

        os.makedirs("data/results", exist_ok=True)
        out_path = f"data/results/day{args.day}_result.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Result saved to: {out_path}")
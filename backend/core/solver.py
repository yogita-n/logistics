from ortools.constraint_solver import routing_enums_pb2, pywrapcp

def solve(locations: list[dict],
          pickup_delivery_pairs: list[tuple],
          time_matrix: list[list[int]],
          soft_window_deadlines: dict,
          soft_penalty: int = 50,
          time_limit_secs: int = 5) -> list[dict] | None:

    n = len(locations)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # n nodes, 1 vehicle, depot=0
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        return time_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    cb_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

    # Time dimension: max 15 min wait, max 120 min route, don't start cumul at zero
    routing.AddDimension(cb_idx, 15, 120, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # Soft time windows — penalise lateness but don't hard-block it
    for node_idx, deadline in soft_window_deadlines.items():
        r_idx = manager.NodeToIndex(node_idx)
        time_dim.SetCumulVarSoftUpperBound(r_idx, deadline, soft_penalty)

    # Pickup-delivery precedence + same vehicle constraint
    for (pickup_node, delivery_node) in pickup_delivery_pairs:
        p_idx = manager.NodeToIndex(pickup_node)
        d_idx = manager.NodeToIndex(delivery_node)
        routing.AddPickupAndDelivery(p_idx, d_idx)
        routing.solver().Add(
            routing.VehicleVar(p_idx) == routing.VehicleVar(d_idx)
        )
        routing.solver().Add(
            time_dim.CumulVar(p_idx) <= time_dim.CumulVar(d_idx)
        )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = time_limit_secs

    solution = routing.SolveWithParameters(params)
    if not solution:
        return None

    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        arrival = solution.Value(time_dim.CumulVar(index))
        route.append({
            "node_idx": node,
            "name": locations[node]["name"],
            "lat": locations[node]["lat"],
            "lng": locations[node]["lng"],
            "arrival_time_mins": arrival
        })
        index = solution.Value(routing.NextVar(index))

    return route

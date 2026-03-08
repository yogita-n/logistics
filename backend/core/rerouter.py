import os
from services.maps_service import build_time_matrix
from services.llm_service import get_rerouting_decision
from core.solver import solve

CONGESTION_THRESHOLD = 0.7

def apply_congestion_penalty(matrix: list[list[int]],
                              congested_nodes: dict,
                              multiplier: float = 2.0) -> list[list[int]]:
    adjusted = [row[:] for row in matrix]
    for node_idx in congested_nodes:
        for j in range(len(matrix)):
            adjusted[node_idx][j] = int(matrix[node_idx][j] * multiplier)
            adjusted[j][node_idx] = int(matrix[j][node_idx] * multiplier)
    return adjusted


def check_and_reroute(agent_state: dict,
                       traffic_scores: dict,
                       locations: list[dict],
                       pairs: list[tuple],
                       deadlines: dict) -> dict:
    """
    agent_state must include: agent_id, current_lat, current_lng,
                              current_route, orders, time_elapsed
    traffic_scores: {node_idx: float} — congestion probability per node
    """
    congested_nodes = {
        n: s for n, s in traffic_scores.items()
        if s > CONGESTION_THRESHOLD
    }

    if not congested_nodes:
        return {
            "rerouted": False,
            "new_route": agent_state["current_route"],
            "reason": "No congestion detected",
            "customer_message": None
        }

    # Rebuild matrix starting from agent's LIVE GPS position (not depot)
    base_matrix = build_time_matrix(
        locations,
        agent_lat=agent_state["current_lat"],
        agent_lng=agent_state["current_lng"]
    )
    adjusted_matrix = apply_congestion_penalty(base_matrix, congested_nodes)

    new_route = solve(locations, pairs, adjusted_matrix, deadlines)

    if new_route is None:
        return {
            "rerouted": False,
            "new_route": agent_state["current_route"],
            "reason": "Reroute solver failed, keeping current route",
            "customer_message": None
        }

    # LLM decides if rerouting is actually worth it
    llm_verdict = get_rerouting_decision(
        agent_state, congested_nodes,
        agent_state["current_route"], new_route
    )

    if llm_verdict.get("should_reroute"):
        return {
            "rerouted": True,
            "new_route": new_route,
            "reason": llm_verdict["reason"],
            "customer_message": llm_verdict["customer_message"]
        }

    return {
        "rerouted": False,
        "new_route": agent_state["current_route"],
        "reason": llm_verdict["reason"],
        "customer_message": None
    }

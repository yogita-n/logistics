const BASE = "http://localhost:5000";

export async function assignRoute(agentId, locations, pairs, deadlines, orders) {
  const res = await fetch(`${BASE}/api/assign-route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      agent_id: agentId,
      locations,
      pickup_delivery_pairs: pairs,
      deadlines,
      orders
    })
  });
  return res.json();
}

export async function sendTrafficScores(agentId, scores) {
  const res = await fetch(`${BASE}/api/traffic-scores`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agent_id: agentId, traffic_scores: scores })
  });
  return res.json();
}

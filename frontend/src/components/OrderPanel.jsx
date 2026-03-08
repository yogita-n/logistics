import { useState } from "react";
import { useAgentRoute } from "../hooks/useWebSocket";
import { assignRoute, sendTrafficScores } from "../api";
import ComparisonView from "./ComparisonView";

const TEST_LOCATIONS = [
  { name: "Depot (Agent Start)",     lat: 12.9250, lng: 77.5938 },
  { name: "Jayanagar Restaurant",    lat: 12.9252, lng: 77.5940 },
  { name: "National College",        lat: 12.9305, lng: 77.5850 },
  { name: "South End Restaurant",    lat: 12.9180, lng: 77.5820 },
  { name: "JP Nagar",                lat: 12.9102, lng: 77.5780 },
  { name: "Lalbagh Restaurant",      lat: 12.9195, lng: 77.5850 },
  { name: "Jayanagar 3rd Block",     lat: 12.9260, lng: 77.5810 },
];
const TEST_PAIRS     = [[1,2],[3,4],[5,6]];
const TEST_DEADLINES = {"2": 25, "4": 30, "6": 30};
const TEST_ORDERS    = ["order_001","order_002","order_003"];

const ORDERS_META = [
  { id: "order_001", pickup: "Jayanagar Restaurant", drop: "National College",    deadline: 25 },
  { id: "order_002", pickup: "South End Restaurant", drop: "JP Nagar",            deadline: 30 },
  { id: "order_003", pickup: "Lalbagh Restaurant",   drop: "Jayanagar 3rd Block", deadline: 30 },
];

export default function OrderPanel({ agentId, route, setRoute }) {
  const [loading, setLoading]   = useState(false);
  const [status, setStatus]     = useState("");
  const [statusType, setType]   = useState("info"); // info | success | error | warning
  const [showComparison, setShowComparison] = useState(false);

  const setMsg = (msg, type = "info") => { setStatus(msg); setType(type); };

  const handleAssignRoute = async () => {
    setLoading(true);
    setMsg("⏳ Running OR-Tools optimisation...", "info");
    try {
      const data = await assignRoute(
        agentId, TEST_LOCATIONS, TEST_PAIRS, TEST_DEADLINES, TEST_ORDERS
      );
      if (data.route) {
        setRoute(data.route);
        setMsg(`✅ Optimised route assigned — ${data.route.length} stops`, "success");
      } else {
        setMsg("❌ Solver returned no route. Try increasing time limit.", "error");
      }
    } catch (e) {
      setMsg(`❌ API Error: ${e.message}`, "error");
    }
    setLoading(false);
  };

  const handleSimulateTraffic = async () => {
    if (!route.length) {
      setMsg("⚠️ Assign a route first before simulating traffic.", "warning");
      return;
    }
    setMsg("🚦 Sending congestion data to rerouting engine...", "info");
    try {
      // Simulates HIGH traffic (0.9) on South End + Lalbagh road nodes
      const result = await sendTrafficScores(agentId, { "3": 0.9, "5": 0.85 });
      if (result.rerouted) {
        setRoute(result.new_route);
        setMsg(`🔄 Rerouted! ${result.reason}`, "warning");
      } else {
        setMsg(`✅ No reroute needed: ${result.reason}`, "success");
      }
    } catch (e) {
      setMsg(`❌ Traffic API Error: ${e.message}`, "error");
    }
  };

  const statusColors = {
    info:    { bg: "#1e3a5f", text: "#7ec8e3" },
    success: { bg: "#1a3d2b", text: "#4caf82" },
    error:   { bg: "#3d1a1a", text: "#ff6b6b" },
    warning: { bg: "#3d2e1a", text: "#f5a623" },
  };
  const sc = statusColors[statusType];

  return (
    <div style={{ padding: 20, background: "#0f0f1a", minHeight: "100%", color: "white",
                  fontFamily: "Inter, sans-serif" }}>

      {/* Header */}
      <h2 style={{ color: "#e94560", margin: "0 0 4px" }}>🚴 Delivery Run</h2>
      <p style={{ color: "#aaa", fontSize: 12, margin: "0 0 20px" }}>
        Agent: <b style={{ color: "white" }}>{agentId}</b>
      </p>

      {/* Action buttons */}
      <button onClick={handleAssignRoute} disabled={loading} style={{
        width: "100%", padding: "11px", marginBottom: 10, borderRadius: 7,
        background: loading ? "#555" : "#e94560", color: "white",
        border: "none", cursor: loading ? "not-allowed" : "pointer",
        fontWeight: "bold", fontSize: 14
      }}>
        {loading ? "⏳ Optimising Route..." : "▶ Assign Optimised Route"}
      </button>

      <button onClick={handleSimulateTraffic} style={{
        width: "100%", padding: "11px", marginBottom: 16, borderRadius: 7,
        background: "#f5a623", color: "white", border: "none",
        cursor: "pointer", fontWeight: "bold", fontSize: 14
      }}>
        🚦 Simulate Heavy Traffic
      </button>
          <button
              onClick={() => setShowComparison(true)}
              disabled={!route.length}
              style={{
                  width: "100%", padding: "11px", marginBottom: 16, borderRadius: 7,
                  background: route.length ? "#4caf82" : "#333",
                  color: "white", border: "none",
                  cursor: route.length ? "pointer" : "not-allowed",
                  fontWeight: "bold", fontSize: 14
              }}
          >
              📊 Compare Routes
          </button>
      {/* Status message */}
      {status && (
        <div style={{ background: sc.bg, color: sc.text, padding: "10px 12px",
                      borderRadius: 7, fontSize: 12, marginBottom: 16, lineHeight: 1.5 }}>
          {status}
        </div>
      )}

      {/* Orders list */}
      <h4 style={{ color: "#aaa", margin: "0 0 8px", fontSize: 12, textTransform: "uppercase",
                   letterSpacing: 1 }}>
        Active Orders
      </h4>
      {ORDERS_META.map((order) => (
        <div key={order.id} style={{ background: "#16213e", borderRadius: 8,
                                     padding: "10px 12px", marginBottom: 8 }}>
          <div style={{ fontSize: 11, color: "#e94560", fontWeight: "bold",
                        marginBottom: 4 }}>
            {order.id.toUpperCase()}
          </div>
          <div style={{ fontSize: 12, color: "#ccc" }}>
            📦 <span style={{ color: "#4caf82" }}>{order.pickup}</span>
          </div>
          <div style={{ fontSize: 12, color: "#ccc", marginTop: 2 }}>
            🏠 <span style={{ color: "#ff6b6b" }}>{order.drop}</span>
          </div>
          <div style={{ fontSize: 11, color: "#777", marginTop: 4 }}>
            ⏱ Soft deadline: {order.deadline} mins
          </div>
        </div>
      ))}

      {/* Optimised route stops */}
      {route.length > 0 && (
        <>
          <h4 style={{ color: "#aaa", margin: "16px 0 8px", fontSize: 12,
                       textTransform: "uppercase", letterSpacing: 1 }}>
            Optimised Stop Sequence
          </h4>
          {route.map((stop, idx) => (
            <div key={idx} style={{ display: "flex", justifyContent: "space-between",
                                    background: "#1a2744", borderRadius: 6,
                                    padding: "7px 10px", marginBottom: 5,
                                    fontSize: 12, alignItems: "center" }}>
              <span>
                <span style={{ color: "#e94560", fontWeight: "bold", marginRight: 6 }}>
                  {idx + 1}.
                </span>
                {stop.name}
              </span>
              <span style={{ color: "#f5a623", fontWeight: "bold", whiteSpace: "nowrap",
                             marginLeft: 8 }}>
                +{stop.arrival_time_mins}m
              </span>
            </div>
          ))}
        </>
      )}

      {/* Legend */}
      <div style={{ marginTop: 20, padding: "10px 12px", background: "#16213e",
                    borderRadius: 8, fontSize: 11, color: "#aaa" }}>
        <div style={{ marginBottom: 3 }}>🟢 Green marker = Pickup point</div>
        <div style={{ marginBottom: 3 }}>🔴 Red marker = Delivery point</div>
        <div>🔵 Blue marker = Agent starting position</div>
      </div>
      {showComparison && (
  <ComparisonView
    optimisedRoute={route}
    onClose={() => setShowComparison(false)}
  />
)}
    </div>
  );
}

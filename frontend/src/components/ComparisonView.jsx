import { MapContainer, TileLayer, Marker, Polyline, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const makeIcon = (color) => new L.Icon({
  iconUrl: `https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-${color}.png`,
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png",
  iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34],
});

const TEST_LOCATIONS = [
  { name: "Depot (Agent Start)",  lat: 12.9250, lng: 77.5938 },
  { name: "Jayanagar Restaurant", lat: 12.9252, lng: 77.5940 },
  { name: "National College",     lat: 12.9305, lng: 77.5850 },
  { name: "South End Restaurant", lat: 12.9180, lng: 77.5820 },
  { name: "JP Nagar",             lat: 12.9102, lng: 77.5780 },
  { name: "Lalbagh Restaurant",   lat: 12.9195, lng: 77.5850 },
  { name: "Jayanagar 3rd Block",  lat: 12.9260, lng: 77.5810 },
];

const PICKUP_NODES   = new Set([1, 3, 5]);
const DELIVERY_NODES = new Set([2, 4, 6]);

// Naive FIFO route: depot → p1→d1 → p2→d2 → p3→d3
const NAIVE_ORDER = [0, 1, 2, 3, 4, 5, 6];

function haversineDistance(a, b) {
  const R    = 6371;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLng = ((b.lng - a.lng) * Math.PI) / 180;
  const x    =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((a.lat * Math.PI) / 180) *
    Math.cos((b.lat * Math.PI) / 180) *
    Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
}

function totalDistance(nodeOrder, locations) {
  let dist = 0;
  for (let i = 0; i < nodeOrder.length - 1; i++) {
    dist += haversineDistance(locations[nodeOrder[i]], locations[nodeOrder[i + 1]]);
  }
  return dist.toFixed(2);
}

function totalTime(route) {
  if (!route || route.length === 0) return 0;
  return route[route.length - 1].arrival_time_mins;
}

function SingleMap({ title, polylineColor, nodeOrder, route, labelColor, badge }) {
  const positions = nodeOrder.map(idx => [
    TEST_LOCATIONS[idx].lat,
    TEST_LOCATIONS[idx].lng
  ]);

  const dist = totalDistance(nodeOrder, TEST_LOCATIONS);
  const time = route ? totalTime(route) : "—";

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column",
                  borderRadius: 12, overflow: "hidden",
                  boxShadow: "0 4px 24px rgba(0,0,0,0.4)" }}>

      {/* Header */}
      <div style={{
        background: labelColor, padding: "14px 20px",
        display: "flex", justifyContent: "space-between", alignItems: "center"
      }}>
        <div>
          <div style={{ color: "white", fontWeight: "bold", fontSize: 15 }}>
            {title}
          </div>
          <div style={{ color: "rgba(255,255,255,0.75)", fontSize: 12, marginTop: 2 }}>
            {badge}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ color: "white", fontSize: 13, fontWeight: "bold" }}>
            📍 {dist} km
          </div>
          <div style={{ color: "rgba(255,255,255,0.75)", fontSize: 12 }}>
            ⏱ {time} mins
          </div>
        </div>
      </div>

      {/* Map */}
      <div style={{ flex: 1 }}>
        <MapContainer
          center={[12.9220, 77.5880]}
          zoom={13}
          style={{ height: "100%", width: "100%" }}
          zoomControl={false}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution="© OpenStreetMap"
          />

          {/* Route polyline */}
          {positions.length > 1 && (
            <Polyline
              positions={positions}
              color={polylineColor}
              weight={4}
              opacity={0.85}
              dashArray={polylineColor === "#aaa" ? "8,5" : undefined}
            />
          )}

          {/* Markers */}
          {nodeOrder.map((nodeIdx, order) => {
            const loc  = TEST_LOCATIONS[nodeIdx];
            const icon = nodeIdx === 0
              ? makeIcon("blue")
              : PICKUP_NODES.has(nodeIdx)
              ? makeIcon("green")
              : makeIcon("red");

            const stopData = route?.find(s => s.node_idx === nodeIdx);

            return (
              <Marker key={nodeIdx} position={[loc.lat, loc.lng]} icon={icon}>
                <Popup>
                  <b>{order + 1}. {loc.name}</b>
                  {PICKUP_NODES.has(nodeIdx)   && <div style={{color:"green"}}>📦 Pickup</div>}
                  {DELIVERY_NODES.has(nodeIdx) && <div style={{color:"red"}}>🏠 Delivery</div>}
                  {nodeIdx === 0               && <div style={{color:"blue"}}>🔵 Start</div>}
                  {stopData && <div>ETA: <b>{stopData.arrival_time_mins} mins</b></div>}
                </Popup>
              </Marker>
            );
          })}

          {/* Stop order numbers on the map */}
          {nodeOrder.map((nodeIdx, order) => {
            const loc = TEST_LOCATIONS[nodeIdx];
            return (
              <Marker
                key={`label-${nodeIdx}`}
                position={[loc.lat + 0.0018, loc.lng]}
                icon={L.divIcon({
                  html: `<div style="
                    background:${labelColor};color:white;
                    border-radius:50%;width:20px;height:20px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:11px;font-weight:bold;
                    box-shadow:0 2px 6px rgba(0,0,0,0.4)">
                    ${order + 1}
                  </div>`,
                  iconSize: [20, 20], iconAnchor: [10, 10], className: ""
                })}
              />
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}

export default function ComparisonView({ optimisedRoute, onClose }) {
  // Build optimised node order from solver output
  const optimisedOrder = optimisedRoute.length > 0
    ? optimisedRoute.map(s => s.node_idx)
    : [];

  const naiveDist = totalDistance(NAIVE_ORDER, TEST_LOCATIONS);
  const optDist   = optimisedOrder.length > 0
    ? totalDistance(optimisedOrder, TEST_LOCATIONS)
    : 0;

  const savedKm   = (naiveDist - optDist).toFixed(2);
  const savedTime = optimisedRoute.length > 0
    ? Math.max(0, 45 - totalTime(optimisedRoute))  // 45 = assumed naive total
    : 0;

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 2000,
      background: "rgba(0,0,0,0.85)",
      display: "flex", flexDirection: "column",
      padding: 20, gap: 16
    }}>

      {/* Top bar */}
      <div style={{ display: "flex", justifyContent: "space-between",
                    alignItems: "center" }}>
        <div>
          <h2 style={{ color: "white", margin: 0, fontSize: 20 }}>
            📊 Route Comparison
          </h2>
          <p style={{ color: "#aaa", margin: "4px 0 0", fontSize: 13 }}>
            Naive FIFO order vs OR-Tools optimised batching
          </p>
        </div>
        <button onClick={onClose} style={{
          background: "#333", color: "white", border: "none",
          borderRadius: 8, padding: "8px 18px",
          cursor: "pointer", fontSize: 14, fontWeight: "bold"
        }}>
          ✕ Close
        </button>
      </div>

      {/* Savings banner */}
      {optimisedOrder.length > 0 && (
        <div style={{
          background: "linear-gradient(135deg, #1a3d2b, #16213e)",
          borderRadius: 10, padding: "12px 20px",
          display: "flex", gap: 40, justifyContent: "center"
        }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ color: "#4caf82", fontSize: 22, fontWeight: "bold" }}>
              {savedKm} km
            </div>
            <div style={{ color: "#aaa", fontSize: 12 }}>Distance Saved</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ color: "#4caf82", fontSize: 22, fontWeight: "bold" }}>
              ~{savedTime} mins
            </div>
            <div style={{ color: "#aaa", fontSize: 12 }}>Time Saved</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ color: "#4caf82", fontSize: 22, fontWeight: "bold" }}>
              {((savedKm / naiveDist) * 100).toFixed(0)}%
            </div>
            <div style={{ color: "#aaa", fontSize: 12 }}>Route Efficiency Gain</div>
          </div>
        </div>
      )}

      {/* Side by side maps */}
      <div style={{ display: "flex", gap: 16, flex: 1, minHeight: 0 }}>
        <SingleMap
          title="❌ Unoptimised Route"
          polylineColor="#aaa"
          nodeOrder={NAIVE_ORDER}
          route={null}
          labelColor="#555"
          badge="FIFO order — no batching intelligence"
        />
        <SingleMap
          title="✅ OR-Tools Optimised"
          polylineColor="#4caf82"
          nodeOrder={optimisedOrder.length > 0 ? optimisedOrder : NAIVE_ORDER}
          route={optimisedRoute}
          labelColor="#1a6b3c"
          badge="Batched · Soft windows · Min cost"
        />
      </div>
    </div>
  );
}

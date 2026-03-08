import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Polyline, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import { useRouteSimulator } from "../hooks/useRouteSimulator";
import "leaflet/dist/leaflet.css";

import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

// Fix Webpack/Vite breaking Leaflet default icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl:       markerIcon,
  shadowUrl:     markerShadow,
});

const makeIcon = (color) => new L.Icon({
  iconUrl: `https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-${color}.png`,
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png",
  iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34],
});

const agentIcon = L.divIcon({
  html: `<div style="font-size:28px;line-height:1;filter:drop-shadow(0 2px 4px rgba(0,0,0,0.5))">🚴</div>`,
  iconSize: [32, 32],
  iconAnchor: [16, 16],
  className: ""
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

function MapFollower({ position }) {
  const map = useMap();
  useEffect(() => {
    if (position) map.panTo(position, { animate: true, duration: 1.2 });
  }, [position]);
  return null;
}

export default function MapView({ agentId, route, alert }) {
  const {
    agentPos,
    currentStopIdx,
    completedStops,
    isPlaying,
    play,
    pause,
    reset
  } = useRouteSimulator(route);

  const completedLine = route
    .slice(0, currentStopIdx + 1)
    .map(s => [s.lat, s.lng]);

  const remainingLine = route
    .slice(currentStopIdx)
    .map(s => [s.lat, s.lng]);

  const journeyDone = route.length > 0 && currentStopIdx >= route.length - 1;

  return (
    <div style={{ height: "100%", width: "100%", position: "relative" }}>

      {/* ── Alert banner ── */}
      {alert && (
        <div style={{
          position: "absolute", top: 16, left: "50%",
          transform: "translateX(-50%)", zIndex: 1000,
          background: "#ff4d4f", color: "white",
          padding: "12px 24px", borderRadius: 8,
          fontWeight: "bold", fontSize: 14,
          boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
          whiteSpace: "nowrap", pointerEvents: "none"
        }}>
          ⚠️ {alert}
        </div>
      )}

      {/* ── Simulation controls ── */}
      {route.length > 0 && (
        <div style={{
          position: "absolute", bottom: 30, left: "50%",
          transform: "translateX(-50%)", zIndex: 1000,
          background: "rgba(15,15,26,0.92)", borderRadius: 12,
          padding: "12px 20px", display: "flex", gap: 12,
          alignItems: "center", boxShadow: "0 4px 20px rgba(0,0,0,0.5)"
        }}>
          {journeyDone ? (
            <span style={{ color: "#4caf82", fontWeight: "bold", fontSize: 13 }}>
              ✅ Delivery Complete!
            </span>
          ) : (
            <>
              <button
                onClick={isPlaying ? pause : play}
                style={{
                  padding: "8px 20px", borderRadius: 7, border: "none",
                  background: isPlaying ? "#f5a623" : "#e94560",
                  color: "white", fontWeight: "bold",
                  cursor: "pointer", fontSize: 14
                }}
              >
                {isPlaying ? "⏸ Pause" : "▶ Play"}
              </button>
              <span style={{ color: "#aaa", fontSize: 12 }}>
                Stop {currentStopIdx + 1} / {route.length}
              </span>
            </>
          )}
          <button
            onClick={reset}
            style={{
              padding: "8px 14px", borderRadius: 7, border: "none",
              background: "#333", color: "#aaa",
              cursor: "pointer", fontSize: 13
            }}
          >
            ↺ Reset
          </button>
        </div>
      )}

      {/* ── Map ── */}
      <MapContainer
        center={[12.9250, 77.5938]}
        zoom={14}
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="© OpenStreetMap contributors"
        />

        {/* Auto-pan to follow agent */}
        {agentPos && <MapFollower position={agentPos} />}

        {/* Completed segment — grey solid */}
        {completedLine.length > 1 && (
          <Polyline
            positions={completedLine}
            color="#555"
            weight={4}
            opacity={0.6}
          />
        )}

        {/* Remaining segment — red dashed */}
        {remainingLine.length > 1 && (
          <Polyline
            positions={remainingLine}
            color="#e94560"
            weight={4}
            opacity={0.85}
            dashArray="8, 5"
          />
        )}

        {/* Location markers */}
        {TEST_LOCATIONS.map((loc, idx) => {
          const routeIdx = route.findIndex(s => s.node_idx === idx);
          const isDone   = completedStops.has(routeIdx);

          const icon = idx === 0
            ? makeIcon("blue")
            : isDone
            ? makeIcon("grey")
            : PICKUP_NODES.has(idx)
            ? makeIcon("green")
            : makeIcon("red");

          const stopInRoute = route.find(s => s.node_idx === idx);

          return (
            <Marker key={idx} position={[loc.lat, loc.lng]} icon={icon}>
              <Popup>
                <b>{loc.name}</b>
                {isDone && (
                  <div style={{ color: "grey" }}>✅ Completed</div>
                )}
                {PICKUP_NODES.has(idx) && !isDone && (
                  <div style={{ color: "green" }}>📦 Pickup</div>
                )}
                {DELIVERY_NODES.has(idx) && !isDone && (
                  <div style={{ color: "red" }}>🏠 Delivery</div>
                )}
                {idx === 0 && (
                  <div style={{ color: "blue" }}>🔵 Agent Start</div>
                )}
                {stopInRoute && (
                  <div>ETA: <b>{stopInRoute.arrival_time_mins} mins</b></div>
                )}
              </Popup>
            </Marker>
          );
        })}

        {/* Moving agent marker */}
        {agentPos && (
          <Marker position={agentPos} icon={agentIcon} zIndexOffset={1000}>
            <Popup>
              <b>🚴 {agentId}</b><br />
              Stop {currentStopIdx + 1} of {route.length}
            </Popup>
          </Marker>
        )}
      </MapContainer>
    </div>
  );
}

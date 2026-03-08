import { useEffect, useState } from "react";
import { io } from "socket.io-client";

export function useAgentRoute(agentId) {
  const [alert, setAlert] = useState(null);

  useEffect(() => {
    const s = io("http://localhost:5000", { transports: ["websocket"] });

    s.on("connect", () => {
      console.log("✅ WebSocket connected");
      s.emit("agent_connect", { agent_id: agentId });
    });

    // route_update comes from backend when traffic triggers reroute
    s.on(`route_update_${agentId}`, (data) => {
      console.log("🔄 Reroute received:", data);
      setAlert(data.reason);
      setTimeout(() => setAlert(null), 8000);
    });

    s.on("disconnect", () => console.log("WebSocket disconnected"));

    return () => s.disconnect();
  }, [agentId]);

  return { alert };
}

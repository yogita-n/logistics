import os, time, threading
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from pymongo import MongoClient
from dotenv import load_dotenv
from core.solver import solve
from core.rerouter import check_and_reroute
from services.maps_service import build_time_matrix
import ssl
import certifi
from pymongo.server_api import ServerApi
load_dotenv()

# Add to top of app.py imports:
from flask_cors import CORS



app = Flask(__name__)
# Allow Vite default ports
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://localhost:5174"]}})
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")



mongo = MongoClient(
    os.getenv("MONGO_URI"),
    server_api=ServerApi('1'),          # forces stable Atlas API
    tls=True,
    tlsCAFile=certifi.where(),          # Use certifi for TLS bundle
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=20000,
    socketTimeoutMS=20000
)
db = mongo["delivery_optimizer"]

# In-memory store for active agent states (augments MongoDB)
active_agents = {}   # { agent_id: agent_state_dict }


# ─── REST: Assign initial optimised route ────────────────────────────────────

@app.route("/api/assign-route", methods=["POST"])
def assign_route():
    data = request.json
    agent_id     = data["agent_id"]
    locations    = data["locations"]
    pairs        = [tuple(p) for p in data["pickup_delivery_pairs"]]
    deadlines    = {int(k): v for k, v in data["deadlines"].items()}

    time_matrix = build_time_matrix(locations)
    route = solve(locations, pairs, time_matrix, deadlines)

    if not route:
        return jsonify({"status": "error", "message": "No solution found"}), 400

    agent_state = {
        "agent_id": agent_id,
        "current_lat": locations[0]["lat"],
        "current_lng": locations[0]["lng"],
        "current_node_idx": 0,
        "orders": data["orders"],
        "current_route": route,
        "time_elapsed": 0,
        "locations": locations,
        "pairs": pairs,
        "deadlines": {str(k): v for k, v in deadlines.items()}  # Mongo requires string keys
    }

    active_agents[agent_id] = agent_state
    db.agent_states.update_one(
        {"agent_id": agent_id},
        {"$set": agent_state},
        upsert=True
    )

    return jsonify({"status": "assigned", "route": route})


# ─── REST: Agent pushes live GPS every 30 secs ───────────────────────────────

@app.route("/api/agent-location", methods=["POST"])
def update_agent_location():
    data = request.json
    agent_id = data["agent_id"]

    if agent_id not in active_agents:
        return jsonify({"status": "error", "message": "Agent not found"}), 404

    active_agents[agent_id]["current_lat"] = data["lat"]
    active_agents[agent_id]["current_lng"] = data["lng"]
    active_agents[agent_id]["time_elapsed"] = data["time_elapsed"]

    db.agent_states.update_one(
        {"agent_id": agent_id},
        {"$set": {
            "current_lat": data["lat"],
            "current_lng": data["lng"],
            "time_elapsed": data["time_elapsed"]
        }}
    )
    return jsonify({"status": "updated"})


# ─── REST: Manual traffic score injection (or from your traffic API) ─────────

@app.route("/api/traffic-scores", methods=["POST"])
def receive_traffic_scores():
    data = request.json
    agent_id = data["agent_id"]
    traffic_scores = {int(k): v for k, v in data["traffic_scores"].items()}

    if agent_id not in active_agents:
        return jsonify({"status": "error", "message": "Agent not found"}), 404

    agent_state = active_agents[agent_id]
    result = check_and_reroute(
        agent_state,
        traffic_scores,
        agent_state["locations"],
        agent_state["pairs"],
        {int(k): v for k, v in agent_state["deadlines"].items()}  # Convert keys back to int for solver
    )

    if result["rerouted"]:
        active_agents[agent_id]["current_route"] = result["new_route"]
        db.agent_states.update_one(
            {"agent_id": agent_id},
            {"$set": {"current_route": result["new_route"]}}
        )
        # Push to agent's app in real time
        socketio.emit(f"route_update_{agent_id}", {
            "new_route": result["new_route"],
            "reason": result["reason"],
            "customer_message": result["customer_message"]
        })

    return jsonify(result)


# ─── Background thread: auto-poll traffic every 90 secs ─────────────────────

def background_traffic_monitor():
    """
    In production, replace mock_get_traffic_scores with your 
    actual traffic prediction API / GNN inference call.
    """
    while True:
        time.sleep(90)
        for agent_id, agent_state in list(active_agents.items()):
            traffic_scores = mock_get_traffic_scores(agent_state)
            result = check_and_reroute(
                agent_state,
                traffic_scores,
                agent_state["locations"],
                agent_state["pairs"],
                agent_state["deadlines"]
            )
            if result["rerouted"]:
                active_agents[agent_id]["current_route"] = result["new_route"]
                socketio.emit(f"route_update_{agent_id}", {
                    "new_route": result["new_route"],
                    "reason": result["reason"],
                    "customer_message": result["customer_message"]
                })


def mock_get_traffic_scores(agent_state: dict) -> dict:
    """
    Replace this with: requests.post(TRAFFIC_API_URL, json={...})
    Returns {node_idx: congestion_probability}
    """
    return {i: 0.2 for i in range(len(agent_state["locations"]))}


# ─── WebSocket: Agent app connects ───────────────────────────────────────────

@socketio.on("agent_connect")
def handle_connect(data):
    from flask_socketio import emit
    agent_id = data["agent_id"]
    print(f"[WS] Agent {agent_id} connected")
    emit("connected", {"message": f"Agent {agent_id} live"})


if __name__ == "__main__":
    monitor_thread = threading.Thread(target=background_traffic_monitor, daemon=True)
    monitor_thread.start()
    socketio.run(app, debug=True, port=5000)

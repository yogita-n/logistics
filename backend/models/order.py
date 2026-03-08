from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class Location:
    name: str
    lat: float
    lng: float

@dataclass
class Order:
    order_id: str
    pickup: Location
    delivery: Location
    placed_at: datetime
    deadline_mins: int          # soft window deadline from pickup time
    status: str = "pending"     # pending | picked_up | delivered
    assigned_agent_id: Optional[str] = None

@dataclass
class AgentState:
    agent_id: str
    current_lat: float
    current_lng: float
    current_node_idx: int       # index in the active location list
    orders: list                # list of Order objects currently in hand
    current_route: list         # list of route dicts from solver
    time_elapsed: float         # minutes since shift started
    last_updated: datetime = field(default_factory=datetime.now)

import pandas as pd
import json

def load_day_data(day=1, traffic_scenario="mostlikely"):
    """
    Loads and prepares all data for one day.
    day              : 1 to 9
    traffic_scenario : 'optimistic', 'mostlikely', or 'pessimistic'
    """

    # ── 1. Load orders for this day ──────────────────────────────────────────
    orders_df = pd.read_excel(
        "data/orders/orders.xlsx",
        sheet_name=f"Day {day}"
    )

    # ── 2. Build stops list ──────────────────────────────────────────────────
    # FIX: renamed 'eat'/'lat' to 'earliest_arrival'/'latest_arrival'
    # to avoid collision with geographic latitude convention.
    def get_priority(eat, lat):
        window = lat - eat
        if window <= 60:    return "URGENT"
        elif window <= 120: return "HIGH"
        elif window <= 180: return "MEDIUM"
        else:               return "LOW"

    stops = []
    for _, row in orders_df.iterrows():
        eat = int(row["EAT"])
        lat = int(row["LAT"])
        stops.append({
            "id":               int(row["NODE_ID"]),
            "weight_kg":        float(row["WEIGHT"]),
            "volume_m3":        float(row["VOLUME"]),
            "service_time":     int(row["SERVICE_TIME"]),
            "earliest_arrival": eat,   # FIX: was "eat" — now unambiguous
            "latest_arrival":   lat,   # FIX: was "lat" — no longer clashes with geo-lat
            "priority":         get_priority(eat, lat)
        })

    # ── 3. Load time matrix ──────────────────────────────────────────────────
    time_df = pd.read_excel(
        f"data/time_and_distance_matrices/day_{day}/time_matrix_{traffic_scenario}_{day}.xlsx",
        index_col=0
    )
    time_matrix = time_df.values.astype(int).tolist()

    # ── 4. Load distance matrix ──────────────────────────────────────────────
    dist_df = pd.read_excel(
        f"data/time_and_distance_matrices/day_{day}/distance_matrix_{day}.xlsx",
        index_col=0
    )
    dist_matrix = dist_df.values.tolist()

    return {
        "day":          day,
        "scenario":     traffic_scenario,
        "num_stops":    len(stops),
        "stops":        stops,
        "time_matrix":  time_matrix,
        "dist_matrix":  dist_matrix
    }


# ── Run and verify ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_day_data(day=1, traffic_scenario="mostlikely")

    print(f"✅ Day         : {data['day']}")
    print(f"✅ Scenario    : {data['scenario']}")
    print(f"✅ Total stops : {data['num_stops']}")
    print(f"✅ Matrix size : {len(data['time_matrix'])} x {len(data['time_matrix'][0])}")

    print("\n=== SAMPLE STOPS (first 5) ===")
    for s in data["stops"][:5]:
        print(s)

    print("\n=== PRIORITY DISTRIBUTION ===")
    from collections import Counter
    priorities = Counter(s["priority"] for s in data["stops"])
    for p, count in sorted(priorities.items()):
        print(f"  {p}: {count} stops")

    print("\n=== SANITY CHECKS ===")
    assert all(data["time_matrix"][i][i] == 0 for i in range(len(data["time_matrix"]))), \
        "❌ Time matrix diagonal is not 0!"
    assert len(data["time_matrix"]) == data["num_stops"] + 1, \
        f"❌ Matrix size mismatch! Expected {data['num_stops']+1}, got {len(data['time_matrix'])}"
    assert all(s["earliest_arrival"] < s["latest_arrival"] for s in data["stops"]), \
        "❌ Some stops have earliest_arrival >= latest_arrival!"
    print("✅ All checks passed — data is ready for OR-Tools!")

    import os
    os.makedirs("data/processed", exist_ok=True)
    out_path = f"data/processed/day{data['day']}_{data['scenario']}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ Saved to: {out_path}")
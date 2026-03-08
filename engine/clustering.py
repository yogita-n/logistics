# engine/clustering.py
"""
Geo-Clustering Engine — K-Means + DBSCAN
-----------------------------------------
Groups delivery stops into geographic zones BEFORE OR-Tools runs.
Benefits:
  • Reduces inter-zone travel by assigning each vehicle a contiguous zone
  • Smaller sub-problems solve faster and more optimally in OR-Tools
  • DBSCAN catches outlier stops that shouldn't be grouped (noise points)

Usage:
    from engine.clustering import ClusteringEngine
    engine = ClusteringEngine(n_clusters=5)
    labels, centers = engine.fit(stops, dist_matrix)
    clustered_data   = engine.split_by_cluster(data, labels)
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler


class ClusteringEngine:
    """
    Two-stage clustering:
      Stage 1 — DBSCAN identifies outlier stops (noise) that have unusual
                 distance patterns and should be served last / separately.
      Stage 2 — K-Means partitions the remaining stops into N vehicle zones.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 2,
        random_state: int = 42
    ):
        self.n_clusters      = n_clusters
        self.dbscan_eps      = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.random_state    = random_state
        self.labels_         = None
        self.cluster_centers_= None
        self.outlier_ids_    = []
        self.coords_2d_      = None

    # ── Internal: project distance matrix → 2-D coordinates via MDS ──────────
    def _dist_to_coords(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        MDS projects the NxN distance matrix into 2-D Euclidean space.
        We use only the stop-to-stop sub-matrix (rows/cols 1..N, excluding depot).
        """
        # Exclude depot row/col (index 0)
        sub = dist_matrix[1:, 1:]
        sub = (sub + sub.T) / 2          # symmetrize
        np.fill_diagonal(sub, 0)

        mds = MDS(
            n_components=2, metric=True,
            n_init=1, random_state=self.random_state,
            normalized_stress="auto"
        )
        coords = mds.fit_transform(sub)
        # Normalize to [0, 1] range for DBSCAN eps to be meaningful
        scaler = StandardScaler()
        return scaler.fit_transform(coords)

    # ── Stage 1: DBSCAN outlier detection ────────────────────────────────────
    def _detect_outliers(self, coords: np.ndarray) -> np.ndarray:
        """Returns boolean mask: True = outlier (noise point in DBSCAN)."""
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        db_labels = db.fit_predict(coords)
        return db_labels == -1   # -1 = noise in DBSCAN

    # ── Stage 2: K-Means zone assignment ─────────────────────────────────────
    def _kmeans_cluster(
        self, coords: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs K-Means on non-outlier stops only.
        Returns (labels array for all stops, cluster centers).
        Outlier stops are assigned to the nearest cluster center.
        """
        non_outlier_coords = coords[~mask]
        n = min(self.n_clusters, len(non_outlier_coords))

        km = KMeans(n_clusters=n, random_state=self.random_state, n_init=10)
        km.fit(non_outlier_coords)

        # Assign ALL stops (including outliers) to nearest cluster
        all_labels = km.predict(coords)
        return all_labels, km.cluster_centers_

    # ── Public API ────────────────────────────────────────────────────────────
    def fit(
        self,
        stops: List[Dict],
        dist_matrix: List[List[float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits clustering on the given stops using the distance matrix.

        Parameters
        ----------
        stops       : list of stop dicts (from preprocessed JSON)
        dist_matrix : NxN matrix including depot at index 0

        Returns
        -------
        labels  : np.ndarray shape (n_stops,) — cluster ID per stop (0-indexed)
        centers : np.ndarray shape (n_clusters, 2) — 2-D cluster centroids
        """
        dist_np = np.array(dist_matrix, dtype=float)
        coords  = self._dist_to_coords(dist_np)
        self.coords_2d_ = coords

        outlier_mask = self._detect_outliers(coords)
        self.outlier_ids_ = [
            stops[i]["id"] for i, is_out in enumerate(outlier_mask) if is_out
        ]

        labels, centers = self._kmeans_cluster(coords, outlier_mask)
        self.labels_          = labels
        self.cluster_centers_ = centers

        return labels, centers

    def split_by_cluster(
        self,
        data: Dict,
        labels: np.ndarray
    ) -> List[Dict]:
        """
        Splits the full day data into per-cluster sub-problems.
        Each sub-problem is a valid data dict that optimizer.py can consume.
        The depot row/col is preserved in every sub-matrix.

        Returns list of dicts, one per cluster.
        """
        dist_matrix = np.array(data["dist_matrix"])
        time_matrix = np.array(data["time_matrix"])
        stops       = data["stops"]
        n_clusters  = int(labels.max()) + 1

        cluster_datasets = []
        for c in range(n_clusters):
            # Indices into stops list (0-indexed) for this cluster
            stop_indices = [i for i, lbl in enumerate(labels) if lbl == c]
            if not stop_indices:
                continue

            cluster_stops = [stops[i] for i in stop_indices]
            node_ids      = [0] + [stops[i]["id"] for i in stop_indices]  # 0=depot

            # Extract sub-matrices
            sub_time = time_matrix[np.ix_(node_ids, node_ids)].tolist()
            sub_dist = dist_matrix[np.ix_(node_ids, node_ids)].tolist()

            # Re-map node IDs to 0..M within the sub-problem
            id_map = {old: new for new, old in enumerate(node_ids)}
            remapped_stops = []
            for s in cluster_stops:
                s2 = dict(s)
                s2["id"] = id_map[s["id"]]
                remapped_stops.append(s2)

            cluster_datasets.append({
                "day":          data["day"],
                "scenario":     data["scenario"],
                "cluster_id":   c,
                "num_stops":    len(cluster_stops),
                "stops":        remapped_stops,
                "time_matrix":  sub_time,
                "dist_matrix":  sub_dist,
                "_node_ids":    node_ids,   # for result re-mapping
                "_id_map":      id_map
            })

        return cluster_datasets

    def cluster_summary(self, stops: List[Dict], labels: np.ndarray) -> Dict:
        """Returns a human-readable summary of cluster assignments."""
        from collections import Counter
        summary = {}
        for c in range(int(labels.max()) + 1):
            cluster_stops = [stops[i] for i, lbl in enumerate(labels) if lbl == c]
            priorities    = Counter(s["priority"] for s in cluster_stops)
            total_weight  = sum(s["weight_kg"] for s in cluster_stops)
            summary[f"cluster_{c}"] = {
                "num_stops":     len(cluster_stops),
                "total_weight":  round(total_weight, 2),
                "priorities":    dict(priorities),
                "stop_ids":      [s["id"] for s in cluster_stops],
                "is_outlier_zone": any(
                    s["id"] in self.outlier_ids_ for s in cluster_stops
                )
            }
        summary["outlier_stop_ids"] = self.outlier_ids_
        return summary


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    with open("data/processed/day1_mostlikely.json") as f:
        data = json.load(f)

    engine = ClusteringEngine(n_clusters=5)
    labels, centers = engine.fit(data["stops"], data["dist_matrix"])

    print(f"✅ Clustering complete — {int(labels.max())+1} clusters")
    print(f"   Outlier stops : {engine.outlier_ids_}")

    summary = engine.cluster_summary(data["stops"], labels)
    for k, v in summary.items():
        if k != "outlier_stop_ids":
            print(f"\n  {k}: {v['num_stops']} stops | "
                  f"{v['total_weight']} kg | priorities: {v['priorities']}")

    datasets = engine.split_by_cluster(data, labels)
    print(f"\n✅ Split into {len(datasets)} sub-problems for OR-Tools")
    for d in datasets:
        print(f"   Cluster {d['cluster_id']}: {d['num_stops']} stops | "
              f"matrix {len(d['time_matrix'])}x{len(d['time_matrix'][0])}")
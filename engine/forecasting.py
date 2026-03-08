# engine/forecasting.py
"""
Demand Forecasting Engine — LSTM + Prophet
-------------------------------------------
Forecasts per-zone delivery demand for future days so vehicles can be
pre-positioned before orders arrive.

Two models run in ensemble:
  • Prophet  — captures weekly seasonality, trend, holiday effects
  • LSTM     — captures non-linear temporal patterns across days

Usage:
    from engine.forecasting import DemandForecaster
    fc = DemandForecaster()
    fc.fit(history)          # history: list of {day, zone, num_stops, weight}
    forecast = fc.predict(horizon_days=3)
"""

import numpy as np
import json
import warnings
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Prophet (optional graceful import) ───────────────────────────────────────
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet not installed. Run: pip install prophet")

# ── PyTorch LSTM ──────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Run: pip install torch")


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM Model Definition
# ══════════════════════════════════════════════════════════════════════════════
class LSTMDemandModel(nn.Module if TORCH_AVAILABLE else object):
    """
    Sequence-to-one LSTM that predicts next-day demand from a window of
    past `seq_len` days.
    Input  shape : (batch, seq_len, n_features)
    Output shape : (batch, 1)
    """
    def __init__(self, input_size: int = 2, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # last timestep output


# ══════════════════════════════════════════════════════════════════════════════
#  Main Forecaster
# ══════════════════════════════════════════════════════════════════════════════
class DemandForecaster:
    """
    Ensemble demand forecaster: Prophet (trend/seasonality) + LSTM (nonlinear).
    Falls back gracefully if either library is unavailable.
    """

    def __init__(
        self,
        seq_len: int = 5,        # LSTM lookback window in days
        lstm_epochs: int = 50,
        lstm_lr: float = 1e-3,
        ensemble_weight_prophet: float = 0.5,  # blend ratio
    ):
        self.seq_len   = seq_len
        self.epochs    = lstm_epochs
        self.lr        = lstm_lr
        self.w_prophet = ensemble_weight_prophet
        self.w_lstm    = 1.0 - ensemble_weight_prophet

        self._prophet_models: Dict[str, "Prophet"] = {}
        self._lstm_models:    Dict[str, "LSTMDemandModel"] = {}
        self._scalers:        Dict[str, Tuple[float, float]] = {}  # (mean, std)
        self._history:        Dict[str, List[Dict]] = defaultdict(list)
        self._fitted = False

    # ── Data preparation ──────────────────────────────────────────────────────
    def _build_zone_series(
        self, history: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Groups history records by zone.
        Each record: {day: int, zone: str, num_stops: int, total_weight: float}
        Returns {zone: [{ds: datetime, y: float}, ...]}
        """
        zone_data: Dict[str, List] = defaultdict(list)
        base_date = datetime(2026, 1, 1)
        for rec in history:
            ds = base_date + timedelta(days=rec["day"] - 1)
            zone_data[rec["zone"]].append({
                "ds": ds,
                "y":  float(rec["num_stops"])
            })
        # Sort by date within each zone
        for zone in zone_data:
            zone_data[zone].sort(key=lambda r: r["ds"])
        return zone_data

    def _normalize(self, values: List[float], zone: str) -> np.ndarray:
        arr  = np.array(values, dtype=np.float32)
        mean = arr.mean()
        std  = arr.std() if arr.std() > 0 else 1.0
        self._scalers[zone] = (mean, std)
        return (arr - mean) / std

    def _denormalize(self, values: np.ndarray, zone: str) -> np.ndarray:
        mean, std = self._scalers[zone]
        return values * std + mean

    def _make_sequences(
        self, series: np.ndarray, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates overlapping windows for LSTM training."""
        X, y = [], []
        for i in range(len(series) - seq_len):
            X.append(series[i: i + seq_len])
            y.append(series[i + seq_len])
        return np.array(X), np.array(y)

    # ── Prophet fit ───────────────────────────────────────────────────────────
    def _fit_prophet(self, zone: str, records: List[Dict]):
        if not PROPHET_AVAILABLE:
            return
        if len(records) < 5:
            print(f"  ⚠️  Zone '{zone}' has only {len(records)} day(s) — need ≥5 for Prophet, skipping")
            return
        import pandas as pd
        df = pd.DataFrame(records)
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            interval_width=0.80
        )
        try:
            model.fit(df)
            self._prophet_models[zone] = model
        except Exception as e:
            print(f"  ⚠️  Prophet failed for zone '{zone}': {e} — will use LSTM/naive fallback")

    def _predict_prophet(self, zone: str, horizon: int) -> np.ndarray:
        if not PROPHET_AVAILABLE or zone not in self._prophet_models:
            return None
        import pandas as pd
        model  = self._prophet_models[zone]
        future = model.make_future_dataframe(periods=horizon)
        fc     = model.predict(future)
        return fc["yhat"].values[-horizon:]

    # ── LSTM fit ──────────────────────────────────────────────────────────────
    def _fit_lstm(self, zone: str, values: List[float]):
        if not TORCH_AVAILABLE or len(values) <= self.seq_len + 1:
            return

        norm   = self._normalize(values, zone)
        # Features: [stops_normalized, day_of_week_sin]
        n      = len(norm)
        dow    = np.array([i % 7 for i in range(n)], dtype=np.float32)
        dow_sin = np.sin(2 * np.pi * dow / 7)
        features = np.stack([norm, dow_sin], axis=1)  # (n, 2)

        X, y = self._make_sequences(features, self.seq_len)
        if len(X) == 0:
            return

        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y[:, 0:1])   # only predict stops (feature 0)

        dataset = TensorDataset(X_t, y_t)
        loader  = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)

        model     = LSTMDemandModel(input_size=2, hidden_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        self._lstm_models[zone] = model

    def _predict_lstm(
        self, zone: str, values: List[float], horizon: int
    ) -> Optional[np.ndarray]:
        if not TORCH_AVAILABLE or zone not in self._lstm_models:
            return None

        model   = self._lstm_models[zone]
        norm    = self._normalize(values, zone)
        n       = len(norm)
        dow_sin = np.sin(2 * np.pi * np.array(
            [(n + i) % 7 for i in range(horizon)], dtype=np.float32) / 7)

        preds = []
        window = list(norm[-self.seq_len:])

        model.eval()
        with torch.no_grad():
            for h in range(horizon):
                feat = np.stack(
                    [np.array(window, dtype=np.float32),
                     np.array([dow_sin[h]] * self.seq_len, dtype=np.float32)],
                    axis=1
                )
                x_t = torch.FloatTensor(feat).unsqueeze(0)   # (1, seq_len, 2)
                pred_norm = model(x_t).item()
                preds.append(pred_norm)
                window.pop(0)
                window.append(pred_norm)

        return self._denormalize(np.array(preds), zone)

    # ── Public API ────────────────────────────────────────────────────────────
    def fit(self, history: List[Dict]):
        """
        Trains Prophet + LSTM per zone.

        Parameters
        ----------
        history : list of dicts with keys:
                  {day: int, zone: str, num_stops: int, total_weight: float}
        """
        zone_series = self._build_zone_series(history)

        for zone, records in zone_series.items():
            self._history[zone] = records
            values = [r["y"] for r in records]

            print(f"  🔧 Fitting zone '{zone}' ({len(values)} days of history)...")
            self._fit_prophet(zone, records)
            self._fit_lstm(zone, values)

        self._fitted = True
        print(f"✅ Forecaster fitted on {len(zone_series)} zones")

    def predict(self, horizon_days: int = 3) -> List[Dict]:
        """
        Predicts demand for the next `horizon_days` days per zone.

        Returns
        -------
        list of dicts: {day_offset, zone, predicted_stops,
                        confidence_low, confidence_high, model_used}
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")

        results = []
        for zone, records in self._history.items():
            values       = [r["y"] for r in records]
            prophet_pred = self._predict_prophet(zone, horizon_days)
            lstm_pred    = self._predict_lstm(zone, values, horizon_days)

            for h in range(horizon_days):
                # Ensemble blend
                if prophet_pred is not None and lstm_pred is not None:
                    pred       = (self.w_prophet * prophet_pred[h] +
                                  self.w_lstm    * lstm_pred[h])
                    model_used = "ensemble (Prophet+LSTM)"
                    conf_lo    = max(0, pred * 0.85)
                    conf_hi    = pred * 1.15
                elif prophet_pred is not None:
                    pred       = prophet_pred[h]
                    model_used = "Prophet"
                    conf_lo    = max(0, pred * 0.80)
                    conf_hi    = pred * 1.20
                elif lstm_pred is not None:
                    pred       = lstm_pred[h]
                    model_used = "LSTM"
                    conf_lo    = max(0, pred * 0.85)
                    conf_hi    = pred * 1.15
                else:
                    # Naive fallback: last known value
                    pred       = values[-1] if values else 0
                    model_used = "naive (fallback)"
                    conf_lo    = max(0, pred * 0.70)
                    conf_hi    = pred * 1.30

                results.append({
                    "day_offset":       h + 1,
                    "zone":             zone,
                    "predicted_stops":  round(max(0, pred), 1),
                    "confidence_low":   round(conf_lo, 1),
                    "confidence_high":  round(conf_hi, 1),
                    "model_used":       model_used
                })

        return results

    def build_history_from_results(
        self, result_paths: List[str], cluster_labels_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Convenience: builds history list from existing day result JSONs.
        If no cluster labels are available, treats all stops as zone "all".
        """
        import os
        history = []
        for path in result_paths:
            if not os.path.exists(path):
                continue
            with open(path) as f:
                r = json.load(f)
            day = r["day"]
            for route in r["optimized_routes"]:
                history.append({
                    "day":          day,
                    "zone":         f"vehicle_{route['vehicle_id']}",
                    "num_stops":    route["num_stops"],
                    "total_weight": 0.0   # weight not in result — placeholder
                })
        return history


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, glob

    result_paths = sorted(glob.glob("data/results/day*_result.json"))
    if not result_paths:
        print("❌ No result JSONs found. Run optimizer first.")
    else:
        fc = DemandForecaster(seq_len=3, lstm_epochs=30)
        history = fc.build_history_from_results(result_paths)
        print(f"📚 Built history from {len(result_paths)} days")
        fc.fit(history)
        forecast = fc.predict(horizon_days=3)
        print(f"\n📈 Forecast for next 3 days:")
        for f in forecast[:9]:
            print(f"  Day+{f['day_offset']} | {f['zone']:20s} | "
                  f"{f['predicted_stops']:5.1f} stops "
                  f"[{f['confidence_low']:.1f}–{f['confidence_high']:.1f}] "
                  f"via {f['model_used']}")
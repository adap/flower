"""Global quality trends panel payload builder."""

from __future__ import annotations

from typing import Any


METRICS = ["macro_f1", "event_peak_macro_f1", "anomaly_auroc"]


def build_quality_payload(state: dict[str, Any]) -> dict[str, Any]:
    quality = state.get("quality", {})
    domains_out: dict[str, Any] = {}
    for domain, points in quality.items():
        raw_series = {m: [] for m in METRICS}
        gated_series = {m: [] for m in METRICS}
        for point in points:
            idx = point.get("index")
            ts = point.get("ts_utc")
            raw = point.get("raw", {})
            gated = point.get("gated", {})
            for metric in METRICS:
                raw_series[metric].append({"x": idx, "ts_utc": ts, "y": raw.get(metric)})
                gated_series[metric].append({"x": idx, "ts_utc": ts, "y": gated.get(metric)})
        domains_out[domain] = {
            "raw": raw_series,
            "gated": gated_series,
            "last_unknown_threshold": points[-1].get("unknown_threshold") if points else None,
        }

    return {
        "panel_id": "global_quality_trends",
        "run_name": state.get("run_name"),
        "metrics": METRICS,
        "domains": domains_out,
        "view_modes": ["raw", "gated"],
        "default_view": "gated",
    }

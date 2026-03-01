"""Shared constants for Comcast anomaly FL experiments."""

from __future__ import annotations

V2_CLASS_NAMES = [
    "normal",
    "narrowband_ingress",
    "cpd",
    "micro_reflection",
    "amp_compression",
    "impulse_noise",
    "unknown_other",
]
V2_REGIMES = ["pre_event", "event_peak", "post_event"]
V2_SIGNAL_DOMAINS = ["downstream_rxmer", "upstream_return"]

V2_NODE_CONTEXT_FEATURES = [
    "utilization",
    "snr_margin",
    "split_mhz",
    "total_amps_in_node",
    "amps_in_series_on_leg",
    "leg_isolated",
    "flat_loss_db",
    "amp_nf_db",
    "amp_cin_db",
    "tcp_headroom_db",
    "hour_of_day_sin",
    "hour_of_day_cos",
]

V2_EDGE_TARGETS = {
    "max_params": 200_000,
    "p95_latency_ms_cpu_proxy": 5.0,
    "quantization_ready": True,
}

V2_ACCEPTANCE_TARGETS = {
    "macro_f1_min": 0.70,
    "event_peak_macro_f1_delta_vs_public_min": 0.03,
    "edge_targets_required": True,
}

NUM_BINS = 256
TIME_STEPS = 16
NUM_CLASSES = len(V2_CLASS_NAMES)

DEFAULT_REGIME_MIX = [0.30, 0.45, 0.25]
DEFAULT_CLASS_PROBS_BY_DOMAIN = {
    "downstream_rxmer": [
        [0.64, 0.10, 0.08, 0.07, 0.06, 0.03, 0.02],
        [0.38, 0.16, 0.12, 0.10, 0.09, 0.08, 0.07],
        [0.54, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04],
    ],
    "upstream_return": [
        [0.56, 0.12, 0.10, 0.05, 0.05, 0.07, 0.05],
        [0.28, 0.19, 0.16, 0.07, 0.07, 0.13, 0.10],
        [0.45, 0.14, 0.12, 0.06, 0.06, 0.10, 0.07],
    ],
}

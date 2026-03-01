"""Synthetic v2 Comcast signal generation with controllable non-IID severity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from .config import ExperimentConfig, resolve_non_iid
from .constants import (
    NUM_BINS,
    NUM_CLASSES,
    TIME_STEPS,
    V2_NODE_CONTEXT_FEATURES,
    V2_REGIMES,
)
from .utils import stable_int_seed


FREQ = np.linspace(0.0, 1.0, NUM_BINS, dtype=np.float32)
TIME_AXIS = np.linspace(0.0, 1.0, TIME_STEPS, dtype=np.float32)

REGIME_UTIL_MEAN = np.array([0.58, 0.92, 0.70], dtype=np.float32)
REGIME_SNR_SHIFT = np.array([0.0, -1.8, -0.8], dtype=np.float32)


@dataclass(slots=True)
class ClientProfile:
    regime_mix: np.ndarray
    class_probs_by_regime: np.ndarray
    split_probs: np.ndarray
    context_offsets: dict[str, float]
    template_scale: float
    template_basis_freq: np.ndarray
    template_basis_time: np.ndarray


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    total = float(v.sum())
    if total <= 0.0:
        return np.ones_like(v) / float(v.size)
    return v / total


def _base_split_probs(signal_domain: str) -> np.ndarray:
    if signal_domain == "downstream_rxmer":
        return np.array([0.65, 0.25, 0.10], dtype=np.float32)
    return np.array([0.35, 0.35, 0.30], dtype=np.float32)


def _domain_base_snr(signal_domain: str) -> float:
    return 35.8 if signal_domain == "downstream_rxmer" else 33.0


def make_client_profile(client_id: str, signal_domain: str, cfg: ExperimentConfig) -> ClientProfile:
    """Build deterministic non-IID profile for one client and one domain."""
    seed = stable_int_seed(cfg.seed, signal_domain, client_id, "profile")
    rng = np.random.default_rng(seed)

    sev = resolve_non_iid(cfg)
    s_class = sev["class_skew"]
    s_regime = sev["regime_skew"]
    s_context = sev["context_skew"]
    s_template = sev["template_skew"]

    alpha_class = 20.0 * (1.0 - s_class) + 0.1 * s_class
    alpha_regime = 15.0 * (1.0 - s_regime) + 0.2 * s_regime

    base_regime = _normalize(np.array(cfg.data.regime_mix, dtype=np.float32))
    regime_draw = rng.dirichlet(np.full(len(V2_REGIMES), alpha_regime, dtype=np.float32)).astype(np.float32)
    regime_mix = _normalize((1.0 - s_regime) * base_regime + s_regime * regime_draw)

    base_class = np.array(cfg.data.class_priors_by_domain[signal_domain], dtype=np.float32)
    class_probs = np.zeros_like(base_class)
    for r in range(len(V2_REGIMES)):
        class_draw = rng.dirichlet(np.full(NUM_CLASSES, alpha_class, dtype=np.float32)).astype(np.float32)
        class_probs[r] = _normalize((1.0 - s_class) * base_class[r] + s_class * class_draw)

    context_scale = 0.1 + 1.4 * s_context
    amp_shift = float(rng.normal(0.0, 10.0 * context_scale))
    loss_shift = float(rng.normal(0.0, 1.2 * context_scale))
    util_shift = float(rng.normal(0.0, 0.12 * context_scale))
    series_shift = float(rng.normal(0.0, 2.5 * context_scale))
    iso_delta = float(rng.normal(0.0, 0.22 * context_scale))
    nf_shift = float(rng.normal(0.0, 0.70 * context_scale))
    cin_shift = float(rng.normal(0.0, 2.5 * context_scale))
    tcp_shift = float(rng.normal(0.0, 1.0 * context_scale))
    snr_shift = float(-0.06 * amp_shift - 0.45 * loss_shift - 1.6 * util_shift + rng.normal(0.0, 0.35 * context_scale))

    split_base = _base_split_probs(signal_domain)
    split_draw = rng.dirichlet(np.full(3, 3.0, dtype=np.float32)).astype(np.float32)
    split_probs = _normalize((1.0 - s_context) * split_base + s_context * split_draw)

    template_scale = float(0.1 + 1.4 * s_template)
    template_basis_freq = (
        0.30 * template_scale
        * np.sin(2.0 * np.pi * rng.uniform(2.0, 8.0) * FREQ + rng.uniform(0.0, 2.0 * np.pi))
    ).astype(np.float32)
    template_basis_time = (
        0.25 * template_scale
        * np.sin(2.0 * np.pi * rng.uniform(1.0, 4.0) * TIME_AXIS + rng.uniform(0.0, 2.0 * np.pi))
    ).astype(np.float32)

    return ClientProfile(
        regime_mix=regime_mix,
        class_probs_by_regime=class_probs,
        split_probs=split_probs,
        context_offsets={
            "utilization": util_shift,
            "total_amps_in_node": amp_shift,
            "amps_in_series_on_leg": series_shift,
            "leg_isolated": iso_delta,
            "flat_loss_db": loss_shift,
            "amp_nf_db": nf_shift,
            "amp_cin_db": cin_shift,
            "tcp_headroom_db": tcp_shift,
            "snr_margin": snr_shift,
        },
        template_scale=template_scale,
        template_basis_freq=template_basis_freq,
        template_basis_time=template_basis_time,
    )


def _sample_hour_by_regime(regime_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hours = np.zeros_like(regime_ids, dtype=np.float32)
    for r in range(len(V2_REGIMES)):
        idx = np.where(regime_ids == r)[0]
        if idx.size == 0:
            continue
        if r == 0:
            h = rng.normal(10.0, 3.0, size=idx.size)
        elif r == 1:
            h = rng.normal(19.0, 2.5, size=idx.size)
        else:
            h = rng.normal(14.0, 3.5, size=idx.size)
        hours[idx] = np.mod(h, 24.0)
    return hours.astype(np.float32)


def _sample_v2_labels_and_meta_impl(
    n_samples: int,
    signal_domain: str,
    rng: np.random.Generator,
    profile: ClientProfile,
) -> dict:
    """Sample labels and node context metadata for one client."""
    regime_ids = rng.choice(len(V2_REGIMES), size=n_samples, p=profile.regime_mix).astype(np.int64)

    y = np.zeros(n_samples, dtype=np.int64)
    for r in range(len(V2_REGIMES)):
        idx = np.where(regime_ids == r)[0]
        if idx.size > 0:
            y[idx] = rng.choice(NUM_CLASSES, size=idx.size, p=profile.class_probs_by_regime[r]).astype(np.int64)

    utilization = np.clip(
        rng.normal(loc=REGIME_UTIL_MEAN[regime_ids], scale=0.05, size=n_samples)
        + profile.context_offsets["utilization"],
        0.20,
        0.99,
    ).astype(np.float32)

    total_amps_in_node = np.clip(
        np.round(rng.normal(loc=np.array([24, 46, 34], dtype=np.float32)[regime_ids], scale=5.0))
        + profile.context_offsets["total_amps_in_node"],
        8,
        96,
    ).astype(np.float32)

    amps_in_series_on_leg = np.clip(
        np.round(rng.normal(loc=np.array([5, 9, 7], dtype=np.float32)[regime_ids], scale=1.3))
        + profile.context_offsets["amps_in_series_on_leg"],
        2,
        16,
    ).astype(np.float32)

    leg_prob = np.clip(
        np.array([0.72, 0.52, 0.64], dtype=np.float32)[regime_ids]
        + profile.context_offsets["leg_isolated"],
        0.02,
        0.98,
    )
    leg_isolated = rng.binomial(1, leg_prob, size=n_samples).astype(np.float32)

    flat_loss_db = np.clip(
        rng.normal(loc=np.array([1.8, 4.1, 2.8], dtype=np.float32)[regime_ids], scale=0.9, size=n_samples)
        + profile.context_offsets["flat_loss_db"],
        0.0,
        9.0,
    ).astype(np.float32)

    amp_nf_db = np.clip(
        rng.normal(6.0, 0.55, size=n_samples) + profile.context_offsets["amp_nf_db"],
        4.5,
        8.5,
    ).astype(np.float32)

    amp_cin_db = np.clip(
        rng.normal(56.0, 2.5, size=n_samples) + profile.context_offsets["amp_cin_db"],
        48.0,
        64.0,
    ).astype(np.float32)

    split_mhz = rng.choice(
        np.array([204.0, 396.0, 492.0], dtype=np.float32),
        size=n_samples,
        p=profile.split_probs,
    ).astype(np.float32)

    tcp_headroom_db = np.clip(
        rng.normal(
            loc=np.array([4.8, 2.5, 3.6], dtype=np.float32)[regime_ids]
            - 0.08 * np.maximum(0.0, amps_in_series_on_leg - 6.0)
            + profile.context_offsets["tcp_headroom_db"],
            scale=0.85,
            size=n_samples,
        ),
        0.0,
        8.0,
    ).astype(np.float32)

    hour = _sample_hour_by_regime(regime_ids, rng)
    hour_of_day_sin = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    hour_of_day_cos = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)

    snr_margin = (
        _domain_base_snr(signal_domain)
        + REGIME_SNR_SHIFT[regime_ids]
        - 2.5 * (utilization - 0.5)
        - 0.045 * np.maximum(0.0, total_amps_in_node - 20.0)
        - 0.25 * flat_loss_db
        + 0.15 * leg_isolated
        + profile.context_offsets["snr_margin"]
        + rng.normal(0.0, 0.9, size=n_samples)
    ).astype(np.float32)

    return {
        "signal_domain": signal_domain,
        "y": y,
        "regime_ids": regime_ids,
        "utilization": utilization,
        "snr_margin": snr_margin,
        "split_mhz": split_mhz,
        "total_amps_in_node": total_amps_in_node,
        "amps_in_series_on_leg": amps_in_series_on_leg,
        "leg_isolated": leg_isolated,
        "flat_loss_db": flat_loss_db,
        "amp_nf_db": amp_nf_db,
        "amp_cin_db": amp_cin_db,
        "tcp_headroom_db": tcp_headroom_db,
        "hour_of_day_sin": hour_of_day_sin,
        "hour_of_day_cos": hour_of_day_cos,
    }


def build_v2_node_context(meta: dict) -> np.ndarray:
    cols = []
    for key in V2_NODE_CONTEXT_FEATURES:
        if key not in meta:
            raise KeyError(f"Missing context feature: {key}")
        cols.append(meta[key].astype(np.float32))
    return np.stack(cols, axis=1).astype(np.float32)


def _clamp_domain(x: np.ndarray, domain: str) -> np.ndarray:
    if domain == "downstream_rxmer":
        return np.clip(x, 16.0, 45.0).astype(np.float32)
    return np.clip(x, 10.0, 42.0).astype(np.float32)


def _build_v2_static_profiles_impl(
    meta: dict,
    rng: np.random.Generator,
    profile: ClientProfile,
) -> tuple[np.ndarray, dict]:
    y = meta["y"]
    regime_ids = meta["regime_ids"]
    utilization = meta["utilization"]
    snr_margin = meta["snr_margin"]
    domain = meta["signal_domain"]

    n = y.shape[0]
    tilt = rng.uniform(-1.4, 1.4, size=n).astype(np.float32)

    base = (
        snr_margin[:, None]
        + 0.85 * np.cos(2.0 * np.pi * 1.4 * FREQ)[None, :]
        + 0.30 * np.sin(2.0 * np.pi * 0.55 * FREQ)[None, :]
        + tilt[:, None] * (FREQ[None, :] - 0.5)
    ).astype(np.float32)

    if domain == "upstream_return":
        base -= 0.030 * np.maximum(0.0, meta["total_amps_in_node"] - 20.0)[:, None]
        base -= 0.20 * meta["flat_loss_db"][:, None]
        base += 0.18 * meta["leg_isolated"][:, None]

    impair = np.zeros_like(base, dtype=np.float32)
    latent = {
        "ingress_center": np.zeros(n, dtype=np.float32),
        "ingress_width": np.zeros(n, dtype=np.float32),
        "ingress_depth": np.zeros(n, dtype=np.float32),
        "cpd_freq": np.zeros(n, dtype=np.float32),
        "cpd_phase": np.zeros(n, dtype=np.float32),
        "cpd_amp": np.zeros(n, dtype=np.float32),
        "micro_freq": np.zeros(n, dtype=np.float32),
        "micro_phase": np.zeros(n, dtype=np.float32),
        "micro_amp": np.zeros(n, dtype=np.float32),
        "comp_strength": np.zeros(n, dtype=np.float32),
        "impulse_bins": np.full((n, 8), -1, dtype=np.int64),
        "impulse_amp": np.zeros((n, 8), dtype=np.float32),
        "unknown_mode": np.zeros(n, dtype=np.int64),
    }

    ts = profile.template_scale

    ingress_idx = np.where(y == 1)[0]
    if ingress_idx.size > 0:
        c = rng.uniform(0.07, 0.93, size=ingress_idx.size).astype(np.float32)
        w = rng.uniform(0.010, 0.040, size=ingress_idx.size).astype(np.float32)
        depth = rng.uniform(2.8, 10.5, size=ingress_idx.size).astype(np.float32)
        depth *= np.clip(rng.normal(1.0, 0.20 * ts, size=ingress_idx.size), 0.4, 2.2).astype(np.float32)
        notch = -depth[:, None] * np.exp(-0.5 * ((FREQ[None, :] - c[:, None]) / w[:, None]) ** 2)
        impair[ingress_idx] += notch.astype(np.float32)
        latent["ingress_center"][ingress_idx] = c
        latent["ingress_width"][ingress_idx] = w
        latent["ingress_depth"][ingress_idx] = depth

    cpd_idx = np.where(y == 2)[0]
    if cpd_idx.size > 0:
        h = rng.uniform(16.0, 42.0, size=cpd_idx.size).astype(np.float32)
        p = rng.uniform(0.0, 2.0 * np.pi, size=cpd_idx.size).astype(np.float32)
        a = rng.uniform(1.6, 5.2, size=cpd_idx.size).astype(np.float32)
        a *= np.clip(rng.normal(1.0, 0.24 * ts, size=cpd_idx.size), 0.35, 2.5).astype(np.float32)
        comb = -a[:, None] * (0.5 + 0.5 * np.sin(2.0 * np.pi * h[:, None] * FREQ[None, :] + p[:, None])) ** 2
        impair[cpd_idx] += comb.astype(np.float32)
        latent["cpd_freq"][cpd_idx] = h
        latent["cpd_phase"][cpd_idx] = p
        latent["cpd_amp"][cpd_idx] = a

    micro_idx = np.where(y == 3)[0]
    if micro_idx.size > 0:
        rf = rng.uniform(6.0, 18.0, size=micro_idx.size).astype(np.float32)
        rp = rng.uniform(0.0, 2.0 * np.pi, size=micro_idx.size).astype(np.float32)
        ra = rng.uniform(1.0, 3.6, size=micro_idx.size).astype(np.float32)
        ra *= np.clip(rng.normal(1.0, 0.20 * ts, size=micro_idx.size), 0.4, 2.4).astype(np.float32)
        ripple = -ra[:, None] * np.sin(2.0 * np.pi * rf[:, None] * FREQ[None, :] + rp[:, None])
        impair[micro_idx] += ripple.astype(np.float32)
        latent["micro_freq"][micro_idx] = rf
        latent["micro_phase"][micro_idx] = rp
        latent["micro_amp"][micro_idx] = ra

    comp_idx = np.where(y == 4)[0]
    if comp_idx.size > 0:
        s = rng.uniform(2.2, 5.8, size=comp_idx.size).astype(np.float32)
        s *= np.clip(rng.normal(1.0, 0.18 * ts, size=comp_idx.size), 0.55, 2.1).astype(np.float32)
        shaping = (0.56 + 0.78 * (FREQ[None, :] ** 1.6)).astype(np.float32)
        impair[comp_idx] += -s[:, None] * shaping
        latent["comp_strength"][comp_idx] = s

    impulse_idx = np.where(y == 5)[0]
    if impulse_idx.size > 0:
        for ii in impulse_idx:
            k = rng.integers(3, 9)
            bins = rng.choice(NUM_BINS, size=int(k), replace=False)
            amps = rng.uniform(2.8, 8.0, size=int(k)).astype(np.float32)
            amps *= np.clip(rng.normal(1.0, 0.22 * ts, size=int(k)), 0.45, 2.4).astype(np.float32)
            for j, b in enumerate(bins):
                impair[ii, b] -= amps[j]
                latent["impulse_bins"][ii, j] = int(b)
                latent["impulse_amp"][ii, j] = amps[j]

    unknown_idx = np.where(y == 6)[0]
    if unknown_idx.size > 0:
        modes = rng.choice(4, size=unknown_idx.size).astype(np.int64)
        latent["unknown_mode"][unknown_idx] = modes
        for loc, mode in zip(unknown_idx, modes):
            if mode == 0:
                c = rng.uniform([0.15, 0.55], [0.40, 0.90]).astype(np.float32)
                w = rng.uniform(0.010, 0.035, size=2).astype(np.float32)
                d = rng.uniform(2.0, 6.5, size=2).astype(np.float32)
                for cc, ww, dd in zip(c, w, d):
                    impair[loc] += -dd * np.exp(-0.5 * ((FREQ - cc) / ww) ** 2)
            elif mode == 1:
                impair[loc] += -2.0 * np.sin(2.0 * np.pi * (5.0 + 9.0 * FREQ) * FREQ + rng.uniform(0, 2 * np.pi))
            elif mode == 2:
                pivot = rng.uniform(0.25, 0.75)
                left = -2.2 * (FREQ - pivot)
                right = -5.0 * np.maximum(0.0, FREQ - pivot)
                impair[loc] += left + right
            else:
                impair[loc] += -1.2 * np.sin(2.0 * np.pi * rng.uniform(8.0, 16.0) * FREQ + rng.uniform(0, 2 * np.pi))
                bins = rng.choice(NUM_BINS, size=int(rng.integers(3, 7)), replace=False)
                impair[loc, bins] -= rng.uniform(1.5, 4.5, size=bins.size).astype(np.float32)

    static = base + impair

    # Client-specific shape bias applies to anomalies only.
    anom_mask = (y != 0).astype(np.float32)[:, None]
    static += anom_mask * profile.template_basis_freq[None, :]

    regime_noise = np.array([0.45, 0.95, 0.68], dtype=np.float32)
    if domain == "upstream_return":
        regime_noise = regime_noise + 0.30
    static += rng.normal(0.0, regime_noise[regime_ids][:, None], size=static.shape).astype(np.float32)

    return _clamp_domain(static, domain), latent


def _build_v2_sequence_profiles_impl(
    x_static: np.ndarray,
    meta: dict,
    latent: dict,
    rng: np.random.Generator,
    profile: ClientProfile,
) -> np.ndarray:
    y = meta["y"]
    regime_ids = meta["regime_ids"]
    domain = meta["signal_domain"]

    seq = np.repeat(x_static[:, None, :], TIME_STEPS, axis=1).astype(np.float32)

    util = meta["utilization"]
    split_scale = np.clip((meta["split_mhz"] - 204.0) / 288.0, 0.0, 1.0).astype(np.float32)
    drift_amp = (0.18 + 0.45 * util + 0.25 * split_scale).astype(np.float32)
    global_wave = np.sin(2.0 * np.pi * TIME_AXIS)[None, :, None].astype(np.float32)
    freq_tilt = (FREQ[None, None, :] - 0.5).astype(np.float32)
    seq += drift_amp[:, None, None] * global_wave * freq_tilt

    ingress_idx = np.where(y == 1)[0]
    if ingress_idx.size > 0:
        c = latent["ingress_center"][ingress_idx]
        w = np.maximum(latent["ingress_width"][ingress_idx], 1e-3)
        d = latent["ingress_depth"][ingress_idx]
        ph = rng.uniform(0.0, 2.0 * np.pi, size=ingress_idx.size).astype(np.float32)
        c_t = c[:, None] + 0.020 * np.sin(2.0 * np.pi * 1.8 * TIME_AXIS[None, :] + ph[:, None])
        d_t = d[:, None] * (1.0 + 0.18 * np.sin(2.0 * np.pi * 2.6 * TIME_AXIS[None, :] + ph[:, None]))
        notch = -d_t[:, :, None] * np.exp(-0.5 * ((FREQ[None, None, :] - c_t[:, :, None]) / w[:, None, None]) ** 2)
        seq[ingress_idx] += notch.astype(np.float32)

    cpd_idx = np.where(y == 2)[0]
    if cpd_idx.size > 0:
        h = latent["cpd_freq"][cpd_idx]
        p = latent["cpd_phase"][cpd_idx]
        a = latent["cpd_amp"][cpd_idx]
        amp_t = a[:, None] * (1.0 + 0.24 * np.sin(2.0 * np.pi * 3.2 * TIME_AXIS[None, :] + p[:, None]))
        comb = -amp_t[:, :, None] * (
            0.5 + 0.5 * np.sin(2.0 * np.pi * h[:, None, None] * FREQ[None, None, :] + p[:, None, None])
        ) ** 2
        seq[cpd_idx] += comb.astype(np.float32)

    micro_idx = np.where(y == 3)[0]
    if micro_idx.size > 0:
        rf = latent["micro_freq"][micro_idx]
        rp = latent["micro_phase"][micro_idx]
        ra = latent["micro_amp"][micro_idx]
        phase_t = rp[:, None] + 2.0 * np.pi * 0.20 * TIME_AXIS[None, :]
        ripple = -ra[:, None, None] * np.sin(2.0 * np.pi * rf[:, None, None] * FREQ[None, None, :] + phase_t[:, :, None])
        seq[micro_idx] += ripple.astype(np.float32)

    comp_idx = np.where(y == 4)[0]
    if comp_idx.size > 0:
        s = latent["comp_strength"][comp_idx]
        burst = (1.0 + 0.33 * np.exp(-0.5 * ((TIME_AXIS - 0.72) / 0.17) ** 2)).astype(np.float32)
        shaping = (0.55 + 0.80 * (FREQ ** 1.6)).astype(np.float32)
        seq[comp_idx] += (-s[:, None, None] * burst[None, :, None] * shaping[None, None, :]).astype(np.float32)

    impulse_idx = np.where(y == 5)[0]
    if impulse_idx.size > 0:
        for ii in impulse_idx:
            burst_count = int(rng.integers(2, 6))
            t_idx = rng.choice(TIME_STEPS, size=burst_count, replace=False)
            bins = latent["impulse_bins"][ii]
            amps = latent["impulse_amp"][ii]
            valid = np.where(bins >= 0)[0]
            if valid.size == 0:
                continue
            for tt in t_idx:
                jitter = rng.uniform(0.8, 1.2)
                seq[ii, tt, bins[valid]] -= jitter * amps[valid]

    unknown_idx = np.where(y == 6)[0]
    if unknown_idx.size > 0:
        for ii in unknown_idx:
            mode = latent["unknown_mode"][ii]
            if mode == 0:
                center = rng.uniform(0.18, 0.82)
                width = rng.uniform(0.01, 0.03)
                for t in range(TIME_STEPS):
                    shift = 0.025 * np.sin(2 * np.pi * t / TIME_STEPS)
                    depth = 2.5 + 1.2 * np.cos(2 * np.pi * 2 * t / TIME_STEPS)
                    seq[ii, t] += -depth * np.exp(-0.5 * ((FREQ - (center + shift)) / width) ** 2)
            elif mode == 1:
                ph = rng.uniform(0, 2 * np.pi)
                for t in range(TIME_STEPS):
                    comb = -1.5 * (0.5 + 0.5 * np.sin(2 * np.pi * (12 + 6 * np.sin(ph + t * 0.3)) * FREQ + ph)) ** 2
                    seq[ii, t] += comb.astype(np.float32)
            elif mode == 2:
                slope = -2.8 * (FREQ - 0.4)
                seq[ii] += slope[None, :]
                for _ in range(int(rng.integers(2, 5))):
                    t0 = int(rng.integers(0, TIME_STEPS))
                    b0 = int(rng.integers(0, NUM_BINS - 4))
                    seq[ii, t0, b0 : b0 + 4] -= rng.uniform(1.5, 4.5)
            else:
                a = rng.normal(0.0, 1.0, size=TIME_STEPS).astype(np.float32)
                b = rng.normal(0.0, 1.0, size=NUM_BINS).astype(np.float32)
                seq[ii] += 0.45 * np.outer(a, b).astype(np.float32)

    # Client-specific temporal basis applies to anomalies.
    anom_mask = (y != 0).astype(np.float32)[:, None, None]
    seq += anom_mask * profile.template_basis_time[None, :, None] * profile.template_basis_freq[None, None, :]

    temporal_sigma = np.array([0.26, 0.68, 0.40], dtype=np.float32)[regime_ids]
    if domain == "upstream_return":
        temporal_sigma += 0.18
    seq += rng.normal(0.0, temporal_sigma[:, None, None], size=seq.shape).astype(np.float32)

    return _clamp_domain(seq, domain)


def _safe_split_indices(
    n_samples: int,
    y: np.ndarray,
    regime_ids: np.ndarray,
    n_train: int,
    n_val: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n_samples)
    strata = (y * len(V2_REGIMES) + regime_ids).astype(np.int64)

    try:
        train_idx, temp_idx = train_test_split(
            idx,
            train_size=n_train,
            random_state=seed,
            stratify=strata,
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(idx, train_size=n_train, random_state=seed, shuffle=True)

    val_fraction_of_temp = n_val / float(n_samples - n_train)
    try:
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_fraction_of_temp,
            random_state=seed,
            stratify=strata[temp_idx],
        )
    except ValueError:
        val_idx, test_idx = train_test_split(temp_idx, train_size=val_fraction_of_temp, random_state=seed, shuffle=True)

    return train_idx, val_idx, test_idx


def generate_client_domain_dataset(client_id: str, signal_domain: str, cfg: ExperimentConfig) -> dict:
    """Generate full train/val/test bundle for one client and domain."""
    profile = make_client_profile(client_id=client_id, signal_domain=signal_domain, cfg=cfg)
    seed = stable_int_seed(cfg.seed, signal_domain, client_id, "dataset")
    rng = np.random.default_rng(seed)

    n_train = int(cfg.data.samples_per_client_train)
    n_val = int(cfg.data.samples_per_client_val)
    n_test = int(cfg.data.samples_per_client_test)
    n_total = n_train + n_val + n_test

    meta_all = _sample_v2_labels_and_meta_impl(
        n_samples=n_total,
        signal_domain=signal_domain,
        rng=rng,
        profile=profile,
    )
    x_static, latent = _build_v2_static_profiles_impl(meta=meta_all, rng=rng, profile=profile)
    x_seq = _build_v2_sequence_profiles_impl(
        x_static=x_static,
        meta=meta_all,
        latent=latent,
        rng=rng,
        profile=profile,
    )
    x_context = build_v2_node_context(meta_all)

    y_all = meta_all["y"]
    regime_all = meta_all["regime_ids"]
    tr, va, te = _safe_split_indices(n_total, y_all, regime_all, n_train, n_val, seed)

    def _split(indices: np.ndarray) -> dict:
        split = {
            "X_static": x_static[indices],
            "X_seq": x_seq[indices],
            "X_context": x_context[indices],
            "y": y_all[indices],
            "regime_ids": regime_all[indices],
        }
        for feat in V2_NODE_CONTEXT_FEATURES:
            split[feat] = meta_all[feat][indices]
        return split

    return {
        "client_id": client_id,
        "signal_domain": signal_domain,
        "profile": profile,
        "all": {
            "X_static": x_static,
            "X_seq": x_seq,
            "X_context": x_context,
            "y": y_all,
            "regime_ids": regime_all,
        },
        "train": _split(tr),
        "val": _split(va),
        "test": _split(te),
    }


def _neutral_profile(signal_domain: str) -> ClientProfile:
    class_probs = np.array(
        [
            [0.64, 0.10, 0.08, 0.07, 0.06, 0.03, 0.02],
            [0.38, 0.16, 0.12, 0.10, 0.09, 0.08, 0.07],
            [0.54, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04],
        ],
        dtype=np.float32,
    )
    if signal_domain == "upstream_return":
        class_probs = np.array(
            [
                [0.56, 0.12, 0.10, 0.05, 0.05, 0.07, 0.05],
                [0.28, 0.19, 0.16, 0.07, 0.07, 0.13, 0.10],
                [0.45, 0.14, 0.12, 0.06, 0.06, 0.10, 0.07],
            ],
            dtype=np.float32,
        )
    return ClientProfile(
        regime_mix=np.array([0.30, 0.45, 0.25], dtype=np.float32),
        class_probs_by_regime=class_probs,
        split_probs=_base_split_probs(signal_domain),
        context_offsets={
            "utilization": 0.0,
            "total_amps_in_node": 0.0,
            "amps_in_series_on_leg": 0.0,
            "leg_isolated": 0.0,
            "flat_loss_db": 0.0,
            "amp_nf_db": 0.0,
            "amp_cin_db": 0.0,
            "tcp_headroom_db": 0.0,
            "snr_margin": 0.0,
        },
        template_scale=0.1,
        template_basis_freq=np.zeros(NUM_BINS, dtype=np.float32),
        template_basis_time=np.zeros(TIME_STEPS, dtype=np.float32),
    )


# Public v2 scaffold signatures.
def sample_v2_labels_and_meta(n_samples: int, signal_domain: str) -> dict:
    rng = np.random.default_rng(42)
    profile = _neutral_profile(signal_domain)
    return _sample_v2_labels_and_meta_impl(n_samples, signal_domain, rng, profile)


def build_v2_static_profiles(meta: dict) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(42)
    profile = _neutral_profile(meta["signal_domain"])
    return _build_v2_static_profiles_impl(meta, rng, profile)


def build_v2_sequence_profiles(x_static: np.ndarray, meta: dict, latent: dict) -> np.ndarray:
    rng = np.random.default_rng(43)
    profile = _neutral_profile(meta["signal_domain"])
    return _build_v2_sequence_profiles_impl(x_static, meta, latent, rng, profile)

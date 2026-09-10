"""
generate_data.py
Synthetic NetFlow/Zeek-style windowed traffic generator.

Produces 5-minute traffic windows over N days with embedded attack events.
Each attack event has a "buildup" phase (recon / slow scanning, gradual
port-entropy drop, rising unique-IP count) BEFORE the actual attack onset
(sharp SYN spike, byte-rate spike). This lets a forecasting model learn to
flag rising risk before the attack itself starts -- which is the entire
point of the SIH pitch ("prediction, not detection").

NOTE: This is synthetic data, not CICIDS2017/CSE-CIC-IDS2018/ToN_IoT.
Those datasets require a manual download from academic hosts that this
sandbox cannot reach. The feature schema and window structure here match
what you'd extract from real NetFlow/Zeek logs, so swapping in the real
datasets later means changing this file only -- the rest of the pipeline
(feature_engineering, model, fusion, evaluation) is dataset-agnostic.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42


def generate_windows(days=10, window_minutes=5, n_incidents=14, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    windows_per_day = int(24 * 60 / window_minutes)
    n_windows = days * windows_per_day
    t = pd.date_range("2026-01-01", periods=n_windows, freq=f"{window_minutes}min")

    # --- baseline benign traffic (diurnal pattern + noise) ---
    hour = np.asarray(t.hour) + np.asarray(t.minute) / 60.0
    diurnal = 0.5 + 0.5 * np.sin((hour - 7) / 24 * 2 * np.pi - np.pi / 2)
    diurnal = np.clip(diurnal, 0.15, 1.0).astype(float)

    flow_rate = 200 + 500 * diurnal + rng.normal(0, 25, n_windows)
    unique_ips = 20 + 60 * diurnal + rng.normal(0, 4, n_windows)
    unique_ports = 15 + 25 * diurnal + rng.normal(0, 3, n_windows)
    port_entropy = 3.2 + 0.5 * diurnal + rng.normal(0, 0.12, n_windows)
    syn_ack_ratio = 0.95 + rng.normal(0, 0.03, n_windows)
    byte_rate = 1.2e6 + 3e6 * diurnal + rng.normal(0, 1.5e5, n_windows)
    inter_arrival_std = 0.08 + rng.normal(0, 0.01, n_windows)

    is_attack_now = np.zeros(n_windows, dtype=int)
    incident_id = np.full(n_windows, -1, dtype=int)
    incident_records = []

    # --- embed attack incidents with buildup + onset ---
    candidate_starts = np.linspace(
        windows_per_day, n_windows - windows_per_day, n_incidents
    ).astype(int)
    candidate_starts = candidate_starts + rng.integers(-15, 15, n_incidents)
    candidate_starts = np.clip(candidate_starts, windows_per_day, n_windows - windows_per_day)

    for idx, onset in enumerate(candidate_starts):
        onset = int(onset)
        buildup_len = int(rng.integers(4, 10))       # 20-50 min of recon buildup
        onset_len = int(rng.integers(3, 8))           # 15-40 min of active attack
        attack_kind = rng.choice(["portscan_ddos", "credential_stuffing", "exfil"])

        buildup_start = max(0, onset - buildup_len)
        buildup_slice = slice(buildup_start, onset)
        onset_slice = slice(onset, min(n_windows, onset + onset_len))

        ramp = np.linspace(0.15, 1.0, onset - buildup_start) if onset > buildup_start else np.array([])

        if attack_kind == "portscan_ddos":
            # recon: rising unique ports/ips, falling entropy (scanning sequential ports)
            unique_ports[buildup_slice] += ramp * 180
            unique_ips[buildup_slice] += ramp * 40
            port_entropy[buildup_slice] -= ramp * 1.8
            inter_arrival_std[buildup_slice] -= ramp * 0.04
            # onset: sharp SYN flood signature
            flow_rate[onset_slice] += 4000
            syn_ack_ratio[onset_slice] += 3.5
            byte_rate[onset_slice] += 2e6
            unique_ips[onset_slice] += 250
        elif attack_kind == "credential_stuffing":
            # recon: many failed auth attempts -> rising flow rate to few dest ports, falling entropy
            flow_rate[buildup_slice] += ramp * 350
            port_entropy[buildup_slice] -= ramp * 1.2
            unique_ips[buildup_slice] += ramp * 15
            # onset: sustained high-rate low-entropy traffic
            flow_rate[onset_slice] += 600
            port_entropy[onset_slice] -= 1.4
            syn_ack_ratio[onset_slice] += 0.6
        else:  # exfil
            # recon: quiet internal scanning, entropy drift, subtle byte-rate creep
            port_entropy[buildup_slice] -= ramp * 0.9
            byte_rate[buildup_slice] += ramp * 8e5
            inter_arrival_std[buildup_slice] += ramp * 0.03
            # onset: large sustained byte-rate spike (data leaving network)
            byte_rate[onset_slice] += 6e6
            flow_rate[onset_slice] += 150

        is_attack_now[onset_slice] = 1
        incident_id[buildup_slice] = idx
        incident_id[onset_slice] = idx
        incident_records.append({
            "incident_id": idx,
            "kind": attack_kind,
            "buildup_start": buildup_start,
            "onset_start": onset,
            "onset_end": min(n_windows, onset + onset_len) - 1,
        })

    port_entropy = np.clip(port_entropy, 0.1, 6.0)
    syn_ack_ratio = np.clip(syn_ack_ratio, 0.05, None)
    flow_rate = np.clip(flow_rate, 5, None)
    unique_ips = np.clip(unique_ips, 1, None)
    unique_ports = np.clip(unique_ports, 1, None)
    byte_rate = np.clip(byte_rate, 1e4, None)
    inter_arrival_std = np.clip(inter_arrival_std, 0.005, None)

    df = pd.DataFrame({
        "timestamp": t,
        "flow_rate": flow_rate,
        "unique_ips": unique_ips,
        "unique_ports": unique_ports,
        "port_entropy": port_entropy,
        "syn_ack_ratio": syn_ack_ratio,
        "byte_rate": byte_rate,
        "inter_arrival_std": inter_arrival_std,
        "is_attack_now": is_attack_now,
        "incident_id": incident_id,
    })

    incidents = pd.DataFrame(incident_records)
    return df, incidents, window_minutes


if __name__ == "__main__":
    df, incidents, wm = generate_windows()
    df.to_csv("synthetic_flows.csv", index=False)
    incidents.to_csv("incidents.csv", index=False)
    print(f"Generated {len(df)} windows ({wm}-min each), {len(incidents)} incidents")
    print(incidents)

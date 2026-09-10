"""
pipeline_gru.py
PRECURSOR backend pipeline, deep-temporal variant: same statistical signal +
fusion + lead-time evaluation as pipeline.py, but the "learned" half of the
hybrid model is now the real GRU/LSTM sequence forecaster (sequence_model.py)
instead of the windowed-feature XGBoost stand-in.

Run: python3 pipeline_gru.py [--cell gru|lstm] [--epochs 25]
"""
import argparse
import json
import numpy as np
import pandas as pd

from generate_data import generate_windows
from feature_engineering import add_rolling_features, add_statistical_signal, make_future_labels
from sequence_model import SeqConfig, train_sequence_model

# reuse the exact evaluation logic from pipeline.py so GRU and XGBoost runs
# are scored identically and comparably
from pipeline import (
    LEAD_WINDOWS, FUSION_ALPHA, ALERT_THRESHOLD,
    fuse_and_threshold, evaluate_lead_time, false_alarm_rate,
)
from sklearn.metrics import precision_score, recall_score


def build_dataset():
    df, incidents, wm = generate_windows(days=10, n_incidents=14)
    df = add_rolling_features(df, window=6)
    df = add_statistical_signal(df)
    df["future_attack"] = make_future_labels(df, lead_windows=LEAD_WINDOWS)
    return df, incidents, wm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--out", default="results_gru.json")
    args = parser.parse_args()

    df, incidents, wm = build_dataset()

    cfg = SeqConfig(
        seq_len=args.seq_len, hidden_size=args.hidden_size,
        cell_type=args.cell, epochs=args.epochs,
    )
    print(f"Training {args.cell.upper()} sequence forecaster "
          f"(seq_len={cfg.seq_len}, hidden={cfg.hidden_size}, epochs={cfg.epochs})...")
    model, ml_risk, split, auc, importances = train_sequence_model(df, cfg)

    final_risk, is_warning = fuse_and_threshold(df, ml_risk, alpha=FUSION_ALPHA, threshold=ALERT_THRESHOLD)

    y_true = df["future_attack"].values
    test_slice = slice(split, len(df))
    precision = precision_score(y_true[test_slice], is_warning[test_slice], zero_division=0)
    recall = recall_score(y_true[test_slice], is_warning[test_slice], zero_division=0)

    incident_records, detected, avg_lead = evaluate_lead_time(df, is_warning, incidents, wm)
    fa_count, fa_total, fa_rate = false_alarm_rate(df, is_warning)

    timeline = pd.DataFrame({
        "t": df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M").tolist(),
        "flow_rate": df["flow_rate"].round(1).tolist(),
        "port_entropy": df["port_entropy"].round(3).tolist(),
        "syn_ack_ratio": df["syn_ack_ratio"].round(3).tolist(),
        "unique_ips": df["unique_ips"].round(1).tolist(),
        "byte_rate": df["byte_rate"].round(0).tolist(),
        "statistical_risk": df["statistical_risk"].round(4).tolist(),
        "ml_risk": np.round(ml_risk, 4).tolist(),
        "final_risk": np.round(final_risk, 4).tolist(),
        "is_warning": is_warning.tolist(),
        "is_attack_now": df["is_attack_now"].tolist(),
        "incident_id": df["incident_id"].tolist(),
    })

    top_features = sorted(importances.items(), key=lambda kv: -kv[1])[:8]

    results = {
        "config": {
            "model": f"{args.cell}_sequence_forecaster",
            "window_minutes": wm,
            "lead_windows": LEAD_WINDOWS,
            "seq_len": cfg.seq_len,
            "hidden_size": cfg.hidden_size,
            "fusion_alpha": FUSION_ALPHA,
            "alert_threshold": ALERT_THRESHOLD,
            "train_test_split_index": split,
            "n_windows": len(df),
            "n_incidents": len(incidents),
        },
        "metrics": {
            "auc": None if np.isnan(auc) else round(float(auc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "avg_lead_time_min": round(avg_lead, 1),
            "incidents_detected_early": detected,
            "incidents_total": len(incidents),
            "false_alarms": fa_count,
            "false_alarm_rate": round(fa_rate, 5),
        },
        "feature_importance": [{"feature": f, "importance": round(v, 4)} for f, v in top_features],
        "incidents": incident_records,
        "timeline": timeline.to_dict(orient="list"),
    }

    with open(args.out, "w") as f:
        json.dump(results, f)

    print(json.dumps(results["metrics"], indent=2))
    print(f"\nWrote {args.out} ({len(df)} windows, {len(incidents)} incidents, model={args.cell})")
    for r in incident_records:
        print(f"  incident {r['incident_id']:>2} ({r['kind']:<20}) "
              f"lead={r['lead_time_min']:>3} min  early_detected={r['detected_early']}")


if __name__ == "__main__":
    main()

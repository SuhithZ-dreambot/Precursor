"""
pipeline.py
End-to-end: generate data -> engineer features -> train XGBoost forecaster
-> fuse with statistical (EWMA/CUSUM) signal -> threshold -> measure lead
time per incident -> export everything the dashboard needs to results.json.

Run: python3 pipeline.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import xgboost as xgb

from generate_data import generate_windows
from feature_engineering import add_rolling_features, add_statistical_signal, make_future_labels, RAW_COLS

LEAD_WINDOWS = 3          # forecast horizon: predict attack within next 3 windows
WINDOW_MINUTES = 5
FUSION_ALPHA = 0.35        # final_risk = alpha*statistical + (1-alpha)*xgboost
ALERT_THRESHOLD = 0.5


def build_dataset():
    df, incidents, wm = generate_windows(days=10, n_incidents=14)
    df = add_rolling_features(df, window=6)
    df = add_statistical_signal(df)
    df["future_attack"] = make_future_labels(df, lead_windows=LEAD_WINDOWS)
    return df, incidents, wm


def feature_columns():
    cols = []
    for c in RAW_COLS:
        cols += [c, f"{c}_roll_mean", f"{c}_roll_std", f"{c}_delta"]
    cols += ["stat_composite", "stat_ewma", "stat_cusum"]
    return cols


def train_model(df, feat_cols):
    X = df[feat_cols].values
    y = df["future_attack"].values

    # time-ordered split: train on first 70%, test on last 30% (no shuffling --
    # this is a forecasting task, leaking future into train would be cheating)
    split = int(len(df) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=pos_weight, eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba_all = model.predict_proba(X[:, :])[:, 1]
    test_proba = proba_all[split:]
    auc = roc_auc_score(y_test, test_proba) if y_test.sum() > 0 else float("nan")

    importances = dict(zip(feat_cols, model.feature_importances_.tolist()))
    return model, proba_all, split, auc, importances


def fuse_and_threshold(df, ml_risk, alpha=FUSION_ALPHA, threshold=ALERT_THRESHOLD):
    stat_risk = df["statistical_risk"].values
    final_risk = alpha * stat_risk + (1 - alpha) * ml_risk
    is_warning = (final_risk >= threshold).astype(int)
    return final_risk, is_warning


def evaluate_lead_time(df, is_warning, incidents, window_minutes):
    """For each incident, find the first warning after the buildup starts
    and before/at onset. Lead time = onset_time - first_warning_time."""
    records = []
    detected = 0
    for _, inc in incidents.iterrows():
        b0, onset = int(inc["buildup_start"]), int(inc["onset_start"])
        search = is_warning[b0:onset]
        first_hit = np.argmax(search) if search.any() else None
        if search.any():
            first_hit_idx = b0 + first_hit
            lead_time_min = (onset - first_hit_idx) * window_minutes
            detected += 1
        else:
            first_hit_idx = None
            lead_time_min = 0  # no early warning -- caught late or missed
        records.append({
            "incident_id": int(inc["incident_id"]),
            "kind": inc["kind"],
            "buildup_start": b0,
            "onset_start": onset,
            "onset_end": int(inc["onset_end"]),
            "warning_start": int(first_hit_idx) if first_hit_idx is not None else None,
            "lead_time_min": int(lead_time_min),
            "detected_early": bool(search.any()),
        })
    lead_times = [r["lead_time_min"] for r in records if r["detected_early"]]
    avg_lead = float(np.mean(lead_times)) if lead_times else 0.0
    return records, detected, avg_lead


def false_alarm_rate(df, is_warning):
    # warnings that fire outside any incident's buildup->onset window
    in_incident = (df["incident_id"].values >= 0)
    false_alarms = int(((is_warning == 1) & (~in_incident)).sum())
    total_benign_windows = int((~in_incident).sum())
    return false_alarms, total_benign_windows, false_alarms / max(1, total_benign_windows)


def main():
    df, incidents, wm = build_dataset()
    feat_cols = feature_columns()
    model, ml_risk, split, auc, importances = train_model(df, feat_cols)
    final_risk, is_warning = fuse_and_threshold(df, ml_risk)

    y_true = df["future_attack"].values
    test_slice = slice(split, len(df))
    precision = precision_score(y_true[test_slice], is_warning[test_slice], zero_division=0)
    recall = recall_score(y_true[test_slice], is_warning[test_slice], zero_division=0)

    incident_records, detected, avg_lead = evaluate_lead_time(df, is_warning, incidents, wm)
    fa_count, fa_total, fa_rate = false_alarm_rate(df, is_warning)

    # --- export everything the dashboard needs ---
    # Downsample the raw timeline for the browser (every window is 5 min;
    # 2880 points is fine actually, but keep this knob for larger runs)
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
            "window_minutes": wm,
            "lead_windows": LEAD_WINDOWS,
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

    with open("results.json", "w") as f:
        json.dump(results, f)

    print(json.dumps(results["metrics"], indent=2))
    print(f"\nWrote results.json ({len(df)} windows, {len(incidents)} incidents)")
    print(f"Lead time detail:")
    for r in incident_records:
        print(f"  incident {r['incident_id']:>2} ({r['kind']:<20}) "
              f"lead={r['lead_time_min']:>3} min  early_detected={r['detected_early']}")


if __name__ == "__main__":
    main()

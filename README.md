# SPAR-Net Prototype — AI-Based Network Attack Forecasting & Early Warning
1.https://suhithZ.github.io/Precursor/dashboard.html


2.https://claude.ai/artifact/5jTa3zYk3wyF8tCnSAg7yd
🔗 [Live Dashboard](https://suhithz.github.io/Precursor/dashboard.html)
Working prototype for SIH: **"Prediction, not just detection."** Forecasts the
probability of an attack in the *next* traffic window, instead of only flagging
malicious traffic once it has already started.

## What's in here

| File | Role |
|---|---|
| `generate_data.py` | Synthetic NetFlow/Zeek-style traffic generator, 5-min windows, with realistic pre-attack "buildup" phases (port-scan/DDoS, credential stuffing, exfiltration) |
| `feature_engineering.py` | Rolling temporal features (mean/std/delta) + EWMA/CUSUM statistical drift detector |
| `pipeline.py` | Trains the XGBoost forecaster, fuses it with the statistical signal, thresholds, measures lead time per incident, exports `results.json` |
| `dashboard.html` | Self-contained interactive dashboard (no server, no internet required) that visualizes `results.json` — this is the file to open and demo |
| `results.json` | Pipeline output (regenerated each run) |

## Why synthetic data

The real public datasets referenced in the deck (CICIDS2017, CSE-CIC-IDS2018,
ToN_IoT) are hosted on academic servers that aren't reachable from the build
environment used to create this prototype. `generate_data.py` produces data
with the **same feature schema** you'd extract from real NetFlow/Zeek logs —
flow rate, unique IPs/ports, port entropy, SYN/ACK ratio, byte rate,
inter-arrival variance — with attacks that ramp up gradually before onset
(mimicking real recon/scanning behavior), so the forecasting task is genuine:
the model has to catch the *rise* in risk, not just classify traffic that's
already obviously malicious.

**To use real data:** replace `generate_data.py`'s output with a loader that
emits a dataframe with the same columns (`timestamp`, the 7 raw features,
`is_attack_now`, `incident_id`). Nothing else in the pipeline needs to change.

## How the model works

1. **Feature panel** — 5-min windows get rolling mean/std/delta over the last
   30 minutes, so temporal trends survive instead of collapsing into a single
   current-state snapshot.
2. **Statistical signal** — an EWMA baseline + two-sided CUSUM drift detector
   over a composite of port-entropy drop / SYN-ACK rise / port-count rise.
   Fast, interpretable, adapts to changing traffic.
3. **Learned signal** — an XGBoost classifier trained on the rolling feature
   panel to predict "attack starts within the next 3 windows" (not "attack is
   happening now" — that distinction is the whole point). This stands in for
   the GRU/LSTM sequence model in the full design; the rolling-window features
   give it temporal context without requiring a GPU/PyTorch build step.
4. **Fusion** — `final_risk = 0.35 × statistical_risk + 0.65 × ml_risk`
5. **Threshold + lead time** — risk ≥ 0.5 fires a SOC early-warning alert.
   Lead time = warning timestamp − actual attack onset, measured per incident.

## Running it

```bash
pip install -r requirements.txt
python3 pipeline.py          # regenerates results.json (takes a few seconds)
```

Then rebuild the dashboard with the fresh results:

```bash
python3 - <<'PY'
import json
compact = json.dumps(json.load(open('results.json')), separators=(',',':'))
html = open('dashboard.html').read().replace('__RESULTS_JSON__', compact)
open('dashboard_final.html', 'w').write(html)
PY
```

Open `dashboard_final.html` directly in a browser — no server needed.

(The repo's `dashboard.html` has a `__RESULTS_JSON__` placeholder so you can
regenerate the data without hand-editing the dashboard. The already-built,
ready-to-open file is provided separately.)

## Honest limitations of this prototype

- **Synthetic data.** Metrics (AUC ≈ 0.99, avg lead time ≈ 14 min) reflect how
  well the method works on this synthetic set, not a production accuracy
  claim — same disclaimer as the deck's research slide.
- **XGBoost stands in for GRU/LSTM.** True sequence modeling (recurrent /
  attention-based) is on the roadmap; this sandbox couldn't install a working
  PyTorch build. The rolling-feature XGBoost approach is a legitimate and
  common baseline for this kind of forecasting task, not a placeholder.
- **No adversarial evasion testing, no real-time streaming.** Both called out
  as future work in the deck's feasibility slide.

## Next steps to go from prototype → demo-ready for judges

1. Swap in a real dataset (CICIDS2017 is the easiest first target — it has
   labeled attack timestamps you can use to build real buildup windows).
2. Add a GRU/LSTM head next to XGBoost once you have GPU access, and compare.
3. Wire `dashboard.html` to a live feed (WebSocket) instead of a static
   `results.json` for a live-demo version.

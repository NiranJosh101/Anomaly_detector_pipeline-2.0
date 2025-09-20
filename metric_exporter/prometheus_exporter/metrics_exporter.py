#!/usr/bin/env python3
"""
metrics_exporter.py

Prometheus exporter that exposes:
 - reference metrics (from ensemble_meta.json)
 - live inference metrics (from inference_metrics.json)

Usage:
  python metrics_exporter.py --ref path/to/ensemble_meta.json --live path/to/inference_metrics.json \
    --port 8000 --interval 5 --model_version v1 --env prod
"""

import argparse
import json
import os
import threading
import time
from typing import List

from prometheus_client import Gauge, start_http_server

# ---- Helper utilities ----
def safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load JSON {path}: {e}")
        return None

def ensure_dir_for(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ---- Metric registry: we'll create gauges for both ref and live metrics ----
# We use label 'feature' for per-feature metrics and optional labels model_version/env for all metrics.

def make_gauge(name, doc, labelnames=None):
    if labelnames:
        return Gauge(name, doc, labelnames)
    else:
        return Gauge(name, doc)

# Reference (training) gauges
FEATURE_MEAN_REF = make_gauge("feature_mean_ref", "Reference per-feature mean from training", ["feature", "model_version", "env"])
FEATURE_STD_REF  = make_gauge("feature_std_ref", "Reference per-feature std from training", ["feature", "model_version", "env"])
FEATURE_GLOBAL_MEAN_REF = make_gauge("feature_global_mean_ref", "Global mean across all features/timesteps (training)", ["model_version", "env"])
FEATURE_GLOBAL_STD_REF  = make_gauge("feature_global_std_ref", "Global std across all features/timesteps (training)", ["model_version", "env"])
ERROR_MEDIAN_REF = make_gauge("error_median_ref", "Reference reconstruction error median (validation)", ["model_version", "env"])
ERROR_P95_REF    = make_gauge("error_p95_ref", "Reference reconstruction error 95th percentile (validation)", ["model_version", "env"])
ANOMALY_RATE_REF = make_gauge("anomaly_rate_ref", "Reference anomaly rate from validation", ["model_version", "env"])

# Live (inference) gauges
FEATURE_MEAN_LIVE = make_gauge("feature_mean_live", "Live per-feature mean during inference", ["feature", "model_version", "env"])
FEATURE_STD_LIVE  = make_gauge("feature_std_live", "Live per-feature std during inference", ["feature", "model_version", "env"])
FEATURE_GLOBAL_MEAN_LIVE = make_gauge("feature_global_mean_live", "Live global mean across all features/timesteps", ["model_version", "env"])
FEATURE_GLOBAL_STD_LIVE  = make_gauge("feature_global_std_live", "Live global std across all features/timesteps", ["model_version", "env"])
ERROR_MEDIAN_LIVE = make_gauge("error_median_live", "Live reconstruction error median", ["model_version", "env"])
ERROR_P95_LIVE    = make_gauge("error_p95_live", "Live reconstruction error p95", ["model_version", "env"])
ANOMALY_RATE_LIVE = make_gauge("anomaly_rate_live", "Live anomaly rate during inference", ["model_version", "env"])

# small helper to set labeled gauges for lists of features
def set_per_feature_gauges(gauge, values: List[float], model_version: str, env: str):
    """
    gauge: Gauge object with labels ["feature","model_version","env"]
    values: list of floats (per-feature)
    """
    for i, v in enumerate(values):
        label = f"f{i}"
        try:
            gauge.labels(feature=label, model_version=model_version, env=env).set(float(v))
        except Exception as e:
            print(f"[WARN] Failed set gauge {gauge} for {label}: {e}")

# ---- Loading and applying reference metrics ----
def load_and_set_reference(ref_path, model_version, env):
    meta = safe_load_json(ref_path)
    if meta is None:
        print(f"[ERROR] Reference meta not loaded from {ref_path}")
        return

    # feature_stats may contain 'mean','std','global_mean','global_std'
    fstats = meta.get("feature_stats", {})
    means = fstats.get("mean", [])
    stds  = fstats.get("std", [])

    # set per-feature ref metrics (labelled by index f0,f1,...)
    if isinstance(means, list) and len(means) > 0:
        set_per_feature_gauges(FEATURE_MEAN_REF, means, model_version, env)
    if isinstance(stds, list) and len(stds) > 0:
        set_per_feature_gauges(FEATURE_STD_REF, stds, model_version, env)

    # global summaries
    global_mean = fstats.get("global_mean", None)
    global_std  = fstats.get("global_std", None)
    if global_mean is not None:
        FEATURE_GLOBAL_MEAN_REF.labels(model_version=model_version, env=env).set(float(global_mean))
    if global_std is not None:
        FEATURE_GLOBAL_STD_REF.labels(model_version=model_version, env=env).set(float(global_std))

    # error stats & anomaly rate
    error_stats = meta.get("error_stats", {})
    if error_stats:
        if "median" in error_stats:
            ERROR_MEDIAN_REF.labels(model_version=model_version, env=env).set(float(error_stats["median"]))
        if "p95" in error_stats:
            ERROR_P95_REF.labels(model_version=model_version, env=env).set(float(error_stats["p95"]))

    if "anomaly_rate_ref" in meta:
        ANOMALY_RATE_REF.labels(model_version=model_version, env=env).set(float(meta["anomaly_rate_ref"]))

    print("[INFO] Loaded and set reference metrics from", ref_path)

# ---- Polling loop for live inference metrics ----
def watch_live_metrics(live_path, poll_interval, model_version, env):
    last_mtime = 0.0
    while True:
        try:
            if not os.path.exists(live_path):
                # file not yet written by inference; just wait
                # print once and sleep
                # (keep sleeping; exporter remains up and serving ref metrics)
                # print(f"[DEBUG] live metrics file not found: {live_path}")
                time.sleep(poll_interval)
                continue

            mtime = os.path.getmtime(live_path)
            if mtime == last_mtime:
                time.sleep(poll_interval)
                continue

            last_mtime = mtime
            data = safe_load_json(live_path)
            if data is None:
                time.sleep(poll_interval)
                continue

            # expect same schema as training:
            # {
            #   "feature_stats": {"mean": [...], "std": [...], "global_mean": ..., "global_std": ...},
            #   "error_stats": {"median": ..., "p95": ...},
            #   "anomaly_rate": ...
            # }
            fstats = data.get("feature_stats", {})
            means = fstats.get("mean", [])
            stds  = fstats.get("std", [])
            global_mean = fstats.get("global_mean", None)
            global_std = fstats.get("global_std", None)

            # set per-feature live gauges (update overlapping indices)
            if isinstance(means, list) and len(means) > 0:
                set_per_feature_gauges(FEATURE_MEAN_LIVE, means, model_version, env)
            if isinstance(stds, list) and len(stds) > 0:
                set_per_feature_gauges(FEATURE_STD_LIVE, stds, model_version, env)

            if global_mean is not None:
                FEATURE_GLOBAL_MEAN_LIVE.labels(model_version=model_version, env=env).set(float(global_mean))
            if global_std is not None:
                FEATURE_GLOBAL_STD_LIVE.labels(model_version=model_version, env=env).set(float(global_std))

            err = data.get("error_stats", {})
            if "median" in err:
                ERROR_MEDIAN_LIVE.labels(model_version=model_version, env=env).set(float(err["median"]))
            if "p95" in err:
                ERROR_P95_LIVE.labels(model_version=model_version, env=env).set(float(err["p95"]))

            if "anomaly_rate" in data:
                ANOMALY_RATE_LIVE.labels(model_version=model_version, env=env).set(float(data["anomaly_rate"]))

            print(f"[INFO] Updated live metrics from {live_path} (mtime={mtime})")

        except Exception as e:
            print(f"[ERROR] Exception while polling live metrics: {e}")

        time.sleep(poll_interval)

# ---- CLI / bootstrap ----
def main():
    parser = argparse.ArgumentParser(description="Prometheus exporter for training + inference metrics")
    parser.add_argument("--ref", required=True, help="Path to ensemble_meta.json (training reference)")
    parser.add_argument("--live", required=True, help="Path to latest inference metrics JSON")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port to serve /metrics")
    parser.add_argument("--interval", type=float, default=5.0, help="Poll interval (seconds) to reload live metrics")
    parser.add_argument("--model_version", default="v1", help="model version label")
    parser.add_argument("--env", default="dev", help="environment label (dev/prod/etc)")
    args = parser.parse_args()

    ref_path = args.ref
    live_path = args.live
    port = args.port
    poll_interval = max(1.0, float(args.interval))
    model_version = args.model_version
    env = args.env

    # Start HTTP server for prometheus_client
    start_http_server(port)
    print(f"[INFO] Metrics exporter started on :{port} — scraping not handled here (Prometheus will scrape)")

    # Load & set reference metrics once
    load_and_set_reference(ref_path, model_version, env)

    # Start background thread to poll live metrics file
    t = threading.Thread(target=watch_live_metrics, args=(live_path, poll_interval, model_version, env), daemon=True)
    t.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down exporter.")

if __name__ == "__main__":
    main()

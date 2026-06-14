#!/usr/bin/env python3
"""Benchmark result viewer server.

Loads result.json files from benchmark run directories and provides
a web UI for comparing results across different parameter sets.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, send_file

app = Flask(__name__)

BENCHMARK_DIR = None


def discover_runs(benchmark_dir: str) -> list[dict]:
    """Scan benchmark_dir for result.json files and return metadata."""
    runs = []
    bdir = Path(benchmark_dir)
    if not bdir.exists():
        return runs

    for entry in sorted(bdir.iterdir()):
        result_json = entry / "result.json"
        if not result_json.is_file():
            continue
        try:
            with open(result_json) as f:
                data = json.load(f)
            summary = data.get("summary", {})
            params = data.get("parameters", {})
            iou_stats = summary.get("iou", {})
            time_stats = summary.get("time_ms", {})
            trans_stats = summary.get("translation_error_m", {})
            rot_stats = summary.get("rotation_error_deg", {})

            runs.append({
                "id": entry.name,
                "path": str(entry),
                "timestamp": data.get("timestamp", entry.name),
                "refine_method": data.get("refine_method", "unknown"),
                "params": params,
                "summary": summary,
                "iou_mean": iou_stats.get("mean", 0),
                "iou_median": iou_stats.get("median", 0),
                "iou_min": iou_stats.get("min", 0),
                "iou_max": iou_stats.get("max", 0),
                "iou_std": iou_stats.get("std", 0),
                "time_mean": time_stats.get("mean", 0),
                "time_median": time_stats.get("median", 0),
                "time_total": sum(
                    v.get("time_ms", 0)
                    for v in data.get("results", {}).values()
                    if v.get("success")
                ),
                "trans_mean": trans_stats.get("mean", 0),
                "rot_mean": rot_stats.get("mean", 0),
                "total_samples": summary.get("total_samples", 0),
                "successful": sum(
                    1
                    for v in data.get("results", {}).values()
                    if v.get("iou", 0) >= 0.95
                ),
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: Failed to parse {result_json}: {e}")
    return runs


@app.route("/")
def index():
    runs = discover_runs(BENCHMARK_DIR)
    return render_template("index.html", runs=runs)


@app.route("/api/runs")
def api_runs():
    runs = discover_runs(BENCHMARK_DIR)
    return jsonify(runs)


@app.route("/api/run/<run_id>")
def api_run_detail(run_id):
    run_dir = Path(BENCHMARK_DIR) / run_id
    result_json = run_dir / "result.json"
    if not result_json.is_file():
        return jsonify({"error": "not found"}), 404
    with open(result_json) as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/compare")
def api_compare():
    """Return all runs' per-sample results for side-by-side comparison."""
    runs = discover_runs(BENCHMARK_DIR)
    all_samples = {}
    run_ids = [r["id"] for r in runs]

    for run in runs:
        result_json = Path(run["path"]) / "result.json"
        with open(result_json) as f:
            data = json.load(f)
        for sample_id, sample_data in data.get("results", {}).items():
            if sample_id not in all_samples:
                all_samples[sample_id] = {"sample_id": sample_id}
            all_samples[sample_id][run["id"]] = {
                "iou": sample_data.get("iou"),
                "time_ms": sample_data.get("time_ms"),
                "trans_error": sample_data.get("translation_error_m"),
                "rot_error": sample_data.get("rotation_error_deg"),
                "success": sample_data.get("success", False),
            }

    return jsonify({
        "runs": [{"id": r["id"], "label": f"{r['refine_method']} | cand={r['params'].get('max_candidates', '?')} | {r['timestamp']}"} for r in runs],
        "samples": sorted(all_samples.values(), key=lambda x: x["sample_id"]),
    })


@app.route("/overlay/<run_id>/<path:sample_path>")
def serve_overlay(run_id, sample_path):
    overlay = Path(BENCHMARK_DIR) / run_id / sample_path
    if not overlay.is_file():
        return jsonify({"error": "not found"}), 404
    return send_file(str(overlay), mimetype="image/png")


@app.route("/overlays/<run_id>/<path:sample_path>")
def serve_overlays(run_id, sample_path):
    return serve_overlay(run_id, sample_path)


def main():
    global BENCHMARK_DIR
    parser = argparse.ArgumentParser(description="Benchmark result viewer")
    parser.add_argument(
        "--benchmark-dir",
        default="./benchmark",
        help="Directory containing benchmark run folders (default: ./benchmark)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to bind (default: 8050)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable Flask debug mode"
    )
    args = parser.parse_args()

    BENCHMARK_DIR = os.path.abspath(args.benchmark_dir)
    print(f"Benchmark dir: {BENCHMARK_DIR}")
    runs = discover_runs(BENCHMARK_DIR)
    print(f"Found {len(runs)} benchmark runs")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

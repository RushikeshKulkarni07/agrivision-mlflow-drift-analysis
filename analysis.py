#!/usr/bin/env python3
"""AgriVision AI - data drift analysis using the Population Stability Index (PSI).

Generates a deterministic, synthetic telemetry sample for the AgriVision AI crop
disease detection service (a stable *reference* window and a shifted *current*
window), computes PSI per feature, and writes:

    data/reference.csv
    data/current.csv
    results/drift-report.json

Only the Python standard library is used, so no installation is required.

Usage:
    python analysis.py
    python analysis.py --seed 49 --records 500 --bins 10 --out-dir .
"""
import argparse
import bisect
import csv
import json
import math
import random
import statistics
from pathlib import Path

SEED = 49
RECORDS = 500          # records per window (see noise_floor.py for why not 100)
BINS = 10              # PSI bins (deciles of the reference window)
PSI_MODERATE = 0.10    # 0.10 <= PSI < 0.25  -> moderate change
PSI_SIGNIFICANT = 0.25 # PSI >= 0.25         -> significant drift
EPSILON = 1e-4         # floor for empty bins so ln() is defined

NUMERIC_FEATURES = [
    "image_brightness",
    "image_sharpness",
    "image_size_kb",
    "top1_confidence",
    "inference_latency_ms",
]
BINARY_FIELDS = ["expert_referral", "correct_prediction"]
COLUMNS = ["scan_id"] + NUMERIC_FEATURES + BINARY_FIELDS

# (mean, std, lower clamp, upper clamp) per feature.
# Reference: stable dry-season usage, good lighting, sharp photos.
# Current:   monsoon-season usage, low light, blurrier photos from older phones,
#            lower model confidence and slower inference. image_size_kb is kept
#            unchanged on purpose as a stable control feature.
REFERENCE_PROFILE = {
    "image_brightness":     (128.0, 18.0, 40.0, 230.0),
    "image_sharpness":      (210.0, 40.0, 30.0, 400.0),
    "image_size_kb":        (320.0, 60.0, 80.0, 700.0),
    "top1_confidence":      (0.88, 0.07, 0.30, 0.99),
    "inference_latency_ms": (420.0, 60.0, 150.0, 900.0),
}
CURRENT_PROFILE = {
    "image_brightness":     (104.0, 24.0, 40.0, 230.0),
    "image_sharpness":      (150.0, 45.0, 30.0, 400.0),
    "image_size_kb":        (320.0, 60.0, 80.0, 700.0),
    "top1_confidence":      (0.79, 0.11, 0.30, 0.99),
    "inference_latency_ms": (470.0, 85.0, 150.0, 900.0),
}
REFERRAL_CONFIDENCE_CUTOFF = 0.70  # below this the app asks for expert review


def _clamp(value, low, high):
    return max(low, min(high, value))


def generate_window(rng, profile, records, start_id):
    """Create one window of synthetic scan telemetry."""
    rows = []
    for i in range(records):
        row = {"scan_id": f"S{start_id + i:04d}"}
        for name in NUMERIC_FEATURES:
            mean, std, low, high = profile[name]
            row[name] = round(_clamp(rng.gauss(mean, std), low, high), 4)
        confidence = row["top1_confidence"]
        row["expert_referral"] = int(confidence < REFERRAL_CONFIDENCE_CUTOFF)
        # Expert-verified correctness: more likely when the model is confident.
        # Blurry photos (low sharpness) reduce correctness further.
        p_correct = confidence + 0.06
        if row["image_sharpness"] < 120:
            p_correct -= 0.12
        p_correct = _clamp(p_correct, 0.05, 0.99)
        row["correct_prediction"] = int(rng.random() < p_correct)
        rows.append(row)
    return rows


def population_stability_index(reference, current, bins=BINS):
    """PSI with decile bins taken from the reference window."""
    edges = statistics.quantiles(reference, n=bins, method="inclusive")

    def proportions(values):
        counts = [0] * bins
        for v in values:
            counts[bisect.bisect_right(edges, v)] += 1
        return [max(c / len(values), EPSILON) for c in counts]

    ref_p, cur_p = proportions(reference), proportions(current)
    return sum((c - r) * math.log(c / r) for r, c in zip(ref_p, cur_p))


def classify(psi_value):
    if psi_value >= PSI_SIGNIFICANT:
        return "significant drift"
    if psi_value >= PSI_MODERATE:
        return "moderate change"
    return "low change"


def rate(rows, field):
    return sum(r[field] for r in rows) / len(rows)


def analyse(reference_rows, current_rows, bins=BINS):
    features = {}
    for name in NUMERIC_FEATURES:
        ref = [r[name] for r in reference_rows]
        cur = [r[name] for r in current_rows]
        value = population_stability_index(ref, cur, bins)
        features[name] = {
            "reference_mean": round(statistics.fmean(ref), 4),
            "current_mean": round(statistics.fmean(cur), 4),
            "psi": round(value, 4),
            "status": classify(value),
        }
    rates = {}
    for field in BINARY_FIELDS:
        r, c = rate(reference_rows, field), rate(current_rows, field)
        rates[field] = {
            "reference_rate": round(r, 4),
            "current_rate": round(c, 4),
            "change_percentage_points": round((c - r) * 100, 2),
        }
    drifted = [n for n, f in features.items() if f["psi"] >= PSI_SIGNIFICANT]
    return {
        "features": features,
        "rates": rates,
        "overall_drift_detected": bool(drifted),
        "drifted_features": drifted,
    }


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(seed=SEED, records=RECORDS, bins=BINS, out_dir="."):
    """Generate data, analyse drift and write all output files."""
    out = Path(out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    reference = generate_window(rng, REFERENCE_PROFILE, records, 1)
    current = generate_window(rng, CURRENT_PROFILE, records, records + 1)

    write_csv(out / "data" / "reference.csv", reference)
    write_csv(out / "data" / "current.csv", current)

    report = {
        "project": "AgriVision AI crop disease detection",
        "method": "Population Stability Index (PSI)",
        "parameters": {
            "seed": seed,
            "records_per_window": records,
            "psi_bins": bins,
            "psi_moderate_threshold": PSI_MODERATE,
            "psi_significant_threshold": PSI_SIGNIFICANT,
        },
    }
    report.update(analyse(reference, current, bins))
    with open(out / "results" / "drift-report.json", "w", encoding="utf-8") as h:
        json.dump(report, h, indent=2, sort_keys=True)
        h.write("\n")
    return report


def print_summary(report):
    print("AgriVision AI - data drift analysis (PSI)")
    print(f"{'feature':<22}{'ref mean':>10}{'cur mean':>10}{'PSI':>9}  status")
    for name, f in report["features"].items():
        print(f"{name:<22}{f['reference_mean']:>10.2f}{f['current_mean']:>10.2f}"
              f"{f['psi']:>9.4f}  {f['status']}")
    for field, r in report["rates"].items():
        print(f"{field:<22}{r['reference_rate']:>10.2%}{r['current_rate']:>10.2%}"
              f"{r['change_percentage_points']:>+8.1f}pp")
    verdict = "DETECTED" if report["overall_drift_detected"] else "not detected"
    print(f"Overall drift: {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--records", type=int, default=RECORDS)
    parser.add_argument("--bins", type=int, default=BINS)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    print_summary(run(args.seed, args.records, args.bins, args.out_dir))


if __name__ == "__main__":
    main()

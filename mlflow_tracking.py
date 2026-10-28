#!/usr/bin/env python3
"""Log the AgriVision AI drift analysis to MLflow.

Runs analysis.run(), then records parameters, metrics and artifacts in an
MLflow experiment. By default a local SQLite backend (mlflow.db) is used; set
MLFLOW_TRACKING_URI to log to a different tracking server.

Requires:  pip install -r requirements-mlflow.txt
Usage:     python mlflow_tracking.py
View UI:   mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import os

import mlflow

import analysis

EXPERIMENT_NAME = "agrivision-crop-disease-drift"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    report = analysis.run()  # (re)generates data/ and results/

    with mlflow.start_run(run_name="psi-reference-vs-current") as run:
        mlflow.log_params(report["parameters"])
        mlflow.set_tags({
            "project": "AgriVision AI",
            "drift_type": "data drift (simulated)",
            "method": "PSI",
            "overall_drift_detected": str(report["overall_drift_detected"]),
        })

        for name, f in report["features"].items():
            mlflow.log_metric(f"psi_{name}", f["psi"])
            mlflow.log_metric(f"reference_mean_{name}", f["reference_mean"])
            mlflow.log_metric(f"current_mean_{name}", f["current_mean"])
        for field, r in report["rates"].items():
            mlflow.log_metric(f"reference_rate_{field}", r["reference_rate"])
            mlflow.log_metric(f"current_rate_{field}", r["current_rate"])
            mlflow.log_metric(f"change_pp_{field}", r["change_percentage_points"])
        mlflow.log_metric("drifted_feature_count", len(report["drifted_features"]))
        mlflow.log_metric("overall_drift_detected", int(report["overall_drift_detected"]))

        mlflow.log_artifact("data/reference.csv", artifact_path="data")
        mlflow.log_artifact("data/current.csv", artifact_path="data")
        mlflow.log_artifact("results/drift-report.json", artifact_path="results")

        print(f"Tracking URI : {mlflow.get_tracking_uri()}")
        print(f"Experiment   : {EXPERIMENT_NAME}")
        print(f"Run ID       : {run.info.run_id}")

    logged = mlflow.get_run(run.info.run_id).data.metrics
    print(f"Metrics      : {len(logged)} logged")
    for key in sorted(k for k in logged if k.startswith("psi_")):
        print(f"  {key:<28}{logged[key]:.4f}")


if __name__ == "__main__":
    main()

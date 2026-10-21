"""End-to-end smoke test of the CPU pipeline on a synthetic cache, through the
real command-line entry points (paths redirected with SPACAL_* env vars)."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _run(args, env):
    res = subprocess.run([sys.executable, "-W", "ignore", "-m", *args], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, f"{args}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-4000:]}"
    return res.stdout


def test_pipeline_end_to_end(tmp_path):
    env = {**os.environ,
           "SPACAL_CACHE_ROOT": str(tmp_path / "synthetic_cache"),
           "SPACAL_PREPARED_ROOT": str(tmp_path / "prepared"),
           "SPACAL_RESULTS_ROOT": str(tmp_path / "results")}
    _run(["experiments.make_synthetic_cache", "--n-scenes", "30", "--imgs-per-scene", "3"], env)
    _run(["experiments.prepare_tables", "--all"], env)
    _run(["experiments.run_calibration", "--dataset", "nyu", "--model", "dav2-indoor-large",
          "--n-seeds", "2", "--alphas", "0.1"], env)
    _run(["experiments.e5_shift", "--source", "nyu", "--target", "diode_indoor",
          "--model", "dav2-indoor-large", "--n-seeds", "1", "--scores", "S2-log", "S6-fact-conf"], env)
    _run(["experiments.run_calibration", "--dataset", "nyu", "--model", "dav2-indoor-large",
          "--n-seeds", "2", "--alphas", "0.1", "--raw-only"], env)
    assert "split units" in _run(["experiments.diag_tables"], env)
    _run(["experiments.e11_anchors", "--dataset", "nyu", "--model", "dav2-indoor-large",
          "--ks", "0", "20", "--n-seeds", "1", "--draws", "2"], env)
    e11 = pd.read_csv(tmp_path / "results" / "e11" / "nyu__dav2-indoor-large" / "summary.csv")
    assert {"A-rescale", "A-hybrid", "A-local", "A-both"} <= set(e11.method)
    assert set(e11[e11.method == "A-local"].base) == {"S2-log"}
    assert {"S2-log", "S5-shape"} <= set(e11[e11.method == "A-both"].base)
    _run(["experiments.e11_anchors", "--exp-id", "e13", "--pattern", "cluster", "--dataset", "nyu",
          "--model", "dav2-indoor-large", "--ks", "20", "--n-seeds", "1", "--draws", "1", "--no-oracle",
          "--anchor-dist"], env)
    e13_dir = tmp_path / "results" / "e13" / "nyu__dav2-indoor-large__cluster"
    e13 = pd.read_csv(e13_dir / "summary.csv")
    assert {"A-rescale", "A-hybrid", "A-local", "A-both"} <= set(e13.method)
    dist = pd.read_csv(e13_dir / "dist_summary.csv")
    assert dist.cov_image_mean.between(0, 1).all() and dist.dist_bin.nunique() > 1
    _run(["experiments.e12_safety", "--dataset", "nyu", "--model", "dav2-indoor-large", "--n-seeds", "2"], env)
    e12 = pd.read_csv(tmp_path / "results" / "e12" / "nyu__dav2-indoor-large" / "summary.csv")
    assert {"point", "K1-2s", "K1-lower", "ROI-event", "ROI-event-scene", "task-scalar", "A-both-k20",
            "danger-CRC@1", "danger-CRC-scene@2.5", "task-scalar-scene", "lane-event", "lane-event-scene"} <= set(e12.method)
    assert {"px_cond_miss_img@1", "viol_roi_scene", "viol_lane_scene", "viol_any_px_scene"} <= set(e12.columns)
    assert "code_sha256" in (tmp_path / "results" / "e12" / "nyu__dav2-indoor-large" / "spec.json").read_text()
    assert e12.viol_roi.between(0, 1).all()
    assert "superseded" in _run(["experiments.summarize", "--exp-id", "main"], env)

    out = tmp_path / "results" / "main" / "nyu__dav2-indoor-large"
    assert set(pd.read_csv(out / "runs_s0-1_raw.csv").calibrator) == {"raw"}
    runs = pd.read_csv(out / "runs_s0-1.csv")
    assert set(runs.calibrator) >= {"K0", "K1", "RCPS", "K2", "K2-scene", "K3-depth", "K3-edge", "V4", "V4-scene", "raw"}
    assert runs.seed.nunique() == 2
    conformal = runs[runs.calibrator != "raw"]
    assert conformal.cov_image_mean.between(0, 1).all()
    assert runs[runs.calibrator == "K2-scene"].patch_event_rate_scene.between(0, 1).all()
    assert runs[runs.calibrator == "V4-scene"].v4_rate_scene.between(0, 1).all()
    assert (tmp_path / "results" / "main" / "_summary" / "summary.md").exists()
    assert (tmp_path / "results" / "e5").exists()

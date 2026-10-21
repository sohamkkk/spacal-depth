"""E10: compute cost. Inference timings come from the caches (measured on the
GPU box with CUDA sync); calibration and scale-fit timings come from the
E2-E4 run files. The point: conformal calibration adds milliseconds on
CPU, while TTA / ensembles / gradient uncertainty multiply GPU cost.

    python -m experiments.e10_cost
"""

import glob

import pandas as pd

from experiments.common import results_dir, to_markdown, write_spec
from src.config import RESULTS_ROOT


def main():
    out = results_dir("e10")
    tables = []
    e0 = RESULTS_ROOT / "e0" / "per_image.csv"
    if e0.exists():
        df = pd.read_csv(e0)
        tcols = [c for c in ("t_pred", "t_flip", "t_grad") if c in df]
        inf = df.groupby(["dataset", "model"])[tcols].mean().mul(1000).add_suffix("_ms").reset_index()
        inf.to_csv(out / "inference_ms.csv", index=False)
        tables.append("## Inference (ms / image, GPU)\n\n" + to_markdown(inf))
    else:
        print(f"{e0} missing — run experiments.e0_base_metrics first for inference timings")

    files = glob.glob(str(RESULTS_ROOT / "main" / "*" / "runs_*.csv"))
    if files:
        runs = pd.concat(map(pd.read_csv, files))
        runs = runs[runs.calibrator != "raw"]
        cal = runs.groupby(["dataset", "model", "calibrator"]).calib_seconds.mean().mul(1000).rename("calib_ms").reset_index()
        fit = runs.groupby(["dataset", "model", "score"]).scale_fit_seconds.mean().rename("scale_fit_s").reset_index()
        cal.to_csv(out / "calibration_ms.csv", index=False)
        fit.to_csv(out / "scale_fit_s.csv", index=False)
        tables += ["## Calibration (ms, CPU, whole calibration set)\n\n" + to_markdown(cal),
                   "## Scale-model fit + predict on cal/test (s, CPU)\n\n" + to_markdown(fit)]
    (out / "summary.md").write_text("\n".join(tables))
    write_spec(out, {"experiment": "E10 cost"})
    print("\n".join(tables))


if __name__ == "__main__":
    main()

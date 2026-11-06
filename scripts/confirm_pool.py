"""Confirmatory replication (docs/ANALYSIS_PLAN_2026-11-03.md) as a pool of single-threaded tasks.

One task = one (part, seed, configuration) run; e17 and e18 issue the commands of scripts/run_confirm.sh (same arguments,
same log names, same output dirs). The experiment code is single-threaded numpy, so many tasks run at once.
Memory-aware: a task starts only if the sum over running tasks of max(current RSS, expected peak of its part) plus
the expected peak of the new task stays under MEM_FRAC of the job's memory. The expected peak is --prior-gb until a
task of that part has finished, then PEAK_MARGIN x the largest peak among finished tasks of that part.

Resumable: a task that exits 0 with no Traceback writes logs/confirm_done/task__<log>; the per-seed markers of
slurm/confirm_all.sbatch (logs/confirm_done/<part>_seed<s>) also count as done.
Several pools (jobs, or srun tasks of one multi-node job) may share the queue: a task is claimed
(logs/confirm_done/claim__<log>, holding <job id>.<srun task rank>) before it starts; claims of jobs no longer in
squeue are stale and ignored.
  python -m scripts.confirm_pool --parts e18 e17 --seeds 0-19 --slots 30
e20 is the sensor-error envelope run (analysis-plan amendment of 4 Nov): e18_sensor_dual --envelope, iPad-probe cells.
Stdlib only (no torch).
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

E17_PAIRS = ["kitti:dav2-outdoor-large", "kitti:unidepthv2-large", "kitti:depthpro",
             "nyu:dav2-indoor-large", "nyu:unidepthv2-large", "nyu:depthpro"]
E17_PATTERNS = ["uniform", "rows", "cluster", "edges", "near"]
E18_MODELS = ["unidepthv2-large", "dav2-indoor-large"]
E18_PATTERNS = ["uniform", "cluster"]


def tasks(part, seed, draws, py):
    tag = f"seeds{seed}_{seed}"
    common = ["--seed-start", str(seed), "--n-seeds", "1", "--draws", str(draws)]
    if part == "e17":
        exp = os.environ.get("E17_EXP", "e17_conf")
        for pattern in E17_PATTERNS:
            for pair in E17_PAIRS:
                ds, model = pair.split(":")
                yield (f"{exp}_{ds}__{model}__{pattern}__{tag}",
                       [py, "-W", "ignore", "-m", "experiments.e11_anchors", "--exp-id", exp, "--pattern", pattern,
                        "--dataset", ds, "--model", model, "--ks", "50", "200", "--no-oracle", "--anchor-dist",
                        "--dual", "--dual-rmax", "--lean", "--dual-gamma", "0.5", "--dist-k-random", "20", *common])
    else:
        exp = os.environ.get("E18_EXP", "e18_conf") if part == "e18" else os.environ.get("E20_EXP", "e20_env")
        extra = [] if part == "e18" else ["--envelope", "--cells", "clean/lidar", "lidar/lidar"]
        for pattern in E18_PATTERNS:
            for model in E18_MODELS:
                yield (f"{exp}_arkitscenes__{model}__{pattern}__{tag}",
                       [py, "-W", "ignore", "-m", "experiments.e18_sensor_dual", "--exp-id", exp, "--pattern", pattern,
                        "--dataset", "arkitscenes", "--model", model, "--ks", "200", "--dist-k-random", "20",
                        "--dual-gamma", "0.5", *common, *extra])


def parse_seeds(spec):
    out = []
    for part in spec:
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


PEAK_MARGIN = 1.1
JOB_ID = f"{os.environ.get('SLURM_JOB_ID', f'local{os.getpid()}')}.{os.environ.get('SLURM_PROCID', '0')}"


def live_jobs():
    """Job ids in squeue, or None if squeue is unavailable (then every claim counts as live)."""
    try:
        out = subprocess.run(["squeue", "-h", "-o", "%i"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return set(out.stdout.split()) if out.returncode == 0 else None


def claim(path, live):
    """Atomically claim a task; False if another live job holds it."""
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = path.read_text().strip()
            except FileNotFoundError:
                continue
            if holder == JOB_ID or not holder or live is None or holder.split(".")[0] in live:
                return holder == JOB_ID
            path.unlink(missing_ok=True)
            continue
        os.write(fd, JOB_ID.encode())
        os.close(fd)
        return True
    return False


def rss_gb(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["e18", "e17"], choices=("e17", "e18", "e20"))
    ap.add_argument("--seeds", nargs="+", default=["0-19"], help="e.g. 0-19 or 4 5 6")
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--slots", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--mem-gb", type=float, default=float(os.environ.get("SLURM_MEM_PER_NODE", 0)) / 1024 or 120)
    ap.add_argument("--mem-frac", type=float, default=0.85)
    ap.add_argument("--prior-gb", nargs=2, type=float, default=(8.0, 16.0), metavar=("E18", "E17"),
                    help="initial per-task peak guesses until a task of that part has been observed")
    ap.add_argument("--stagger", type=float, default=20.0, help="seconds between launches (lets RSS show up)")
    args = ap.parse_args()

    py = os.environ.get("PY", sys.executable)
    done_dir = Path("logs/confirm_done")
    done_dir.mkdir(parents=True, exist_ok=True)
    queue = []
    for part in args.parts:
        for seed in parse_seeds(args.seeds):
            if (done_dir / f"{part}_seed{seed}").exists():
                continue
            queue += [(part, name, cmd) for name, cmd in tasks(part, seed, args.draws, py)
                      if not (done_dir / f"task__{name}").exists()]
    prior = {"e18": args.prior_gb[0], "e20": args.prior_gb[0], "e17": args.prior_gb[1]}
    finished_peak = {}  # part -> largest peak among finished tasks
    running_peak = dict.fromkeys(prior, 0.0)

    def expected(part):
        if part in finished_peak:
            return PEAK_MARGIN * finished_peak[part]
        return max(prior[part], running_peak[part])

    budget = args.mem_gb * args.mem_frac
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    live = live_jobs()
    print(f"=== {time.strftime('%FT%T')} job {JOB_ID}: {len(queue)} tasks pending, {args.slots} slots, "
          f"budget {budget:.0f} GB", flush=True)

    running = {}  # pid -> (part, name, Popen, t0, observed peak)
    n_ok = n_fail = n_skip = 0
    last_launch = last_live = 0.0

    def release(name):
        c = done_dir / f"claim__{name}"
        if c.exists() and c.read_text().strip() == JOB_ID:
            c.unlink(missing_ok=True)

    def stop(*_):
        for _, name, p, _, _ in running.values():
            p.terminate()
            release(name)
        sys.exit(1)
    signal.signal(signal.SIGTERM, stop)

    while queue or running:
        for pid in list(running):
            part, name, p, t0, pk = running[pid]
            pk = max(pk, rss_gb(pid))
            running[pid] = (part, name, p, t0, pk)
            running_peak[part] = max(running_peak[part], pk)
            if p.poll() is None:
                continue
            del running[pid]
            log = Path(f"logs/{name}.log")
            ok = p.returncode == 0 and "Traceback" not in log.read_text(errors="ignore")
            mins = (time.time() - t0) / 60
            if ok:
                (done_dir / f"task__{name}").write_text(time.strftime("%FT%T"))
                if pk > 0.5:
                    finished_peak[part] = max(finished_peak.get(part, 0.0), pk)
                n_ok += 1
            else:
                n_fail += 1
            release(name)
            print(f"{'ok  ' if ok else 'FAIL'} {name}  {mins:.0f} min  peak {pk:.1f} GB  "
                  f"[{n_ok} ok, {n_fail} failed, {n_skip} taken by other jobs, {len(running)} running, "
                  f"{len(queue)} queued]", flush=True)
        committed = sum(max(rss_gb(pid), expected(part)) for pid, (part, *_) in running.items())
        if (queue and len(running) < args.slots and time.time() - last_launch >= args.stagger
                and (not running or committed + expected(queue[0][0]) <= budget)):
            part, name, cmd = queue.pop(0)
            if time.time() - last_live > 600:
                live, last_live = live_jobs(), time.time()
            if (done_dir / f"task__{name}").exists() or not claim(done_dir / f"claim__{name}", live):
                n_skip += 1
                continue
            if (done_dir / f"task__{name}").exists():  # finished and released between the check and the claim
                release(name)
                n_skip += 1
                continue
            p = subprocess.Popen(cmd, stdout=open(f"logs/{name}.log", "w"), stderr=subprocess.STDOUT, env=env)
            running[p.pid] = (part, name, p, time.time(), 0.0)
            last_launch = time.time()
            continue
        time.sleep(5)
    print(f"=== {time.strftime('%FT%T')} pass complete: {n_ok} ok, {n_fail} failed, {n_skip} taken by other jobs; "
          "expected peaks " + ", ".join(f"{k} {expected(k):.1f} GB" for k in prior), flush=True)


if __name__ == "__main__":
    main()

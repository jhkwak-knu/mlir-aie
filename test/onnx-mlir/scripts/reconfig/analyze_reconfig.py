"""Analyze the reconfiguration microbenchmark and contextualize against the
measured model totals.

per-switch cost:
    Delta_t(d) = alternating_per_dispatch(d) - mean(baseline_per_dispatch_i)
    Delta_E(d) = same for energy
The saturated (max over d) Delta is the worst-case per-switch reconfiguration
cost, used for a conservative model upper bound.

Model contextualization (no new measurement):
    reconfig_per_inference = N_switch * Delta_saturated
    ratio = reconfig_per_inference / measured_model_total
where N_switch = number of NPU GEMM dispatches per inference (distinct kernel
rows), read from the model's measurements.csv.

Outputs:
    - stdout summary table
    - paper/fig/data/reconfig_v1.csv  (per shape/depth Delta table, figure data)
    - paper/fig/data/reconfig_model_context.csv  (per model/setter ratios)
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ONNX_DIR = Path(__file__).resolve().parents[2]          # test/onnx-mlir
REPO_ROOT = ONNX_DIR.parents[1]                         # mlir-aie
EXPERIMENTS = REPO_ROOT / "experiments"


def load_reconfig(csv_path: Path, run_tag: str | None):
    rows = list(csv.DictReader(csv_path.open()))
    if run_tag:
        return [r for r in rows if r["run_tag"] == run_tag]
    # Each shape is measured in its own run_tag; keep the latest run per shape.
    latest = {}
    for r in rows:
        st = r["shape_tag"]
        latest[st] = max(latest.get(st, ""), r["run_tag"])
    return [r for r in rows if r["run_tag"] == latest[r["shape_tag"]]]


def compute_deltas(rows):
    """Return {(shape_tag, depth): {dt, de, base_t, alt_t, base_e, alt_e}}."""
    by = defaultdict(lambda: {"baseline_t": [], "baseline_e": [],
                              "alt_t": None, "alt_e": None})
    for r in rows:
        key = (r["shape_tag"], int(r["depth"]))
        t = float(r["per_dispatch_us_min"])
        e = float(r["per_dispatch_energy_uj"])
        if r["schedule"] == "baseline":
            by[key]["baseline_t"].append(t)
            by[key]["baseline_e"].append(e)
        else:
            by[key]["alt_t"] = t
            by[key]["alt_e"] = e
    out = {}
    for key, v in sorted(by.items()):
        base_t = statistics.mean(v["baseline_t"]) if v["baseline_t"] else None
        base_e = statistics.mean(v["baseline_e"]) if v["baseline_e"] else None
        dt = (v["alt_t"] - base_t) if (v["alt_t"] is not None and base_t is not None) else None
        de = (v["alt_e"] - base_e) if (v["alt_e"] is not None and base_e is not None and base_e >= 0) else None
        out[key] = {"base_t": base_t, "alt_t": v["alt_t"], "dt": dt,
                    "base_e": base_e, "alt_e": v["alt_e"], "de": de}
    return out


def linear_fit(points):
    """Least-squares slope/intercept for [(cores, dt), ...]."""
    n = len(points)
    sx = sum(c for c, _ in points)
    sy = sum(d for _, d in points)
    mx, my = sx / n, sy / n
    sxx = sum((c - mx) ** 2 for c, _ in points)
    sxy = sum((c - mx) * (d - my) for c, d in points)
    slope = sxy / sxx if sxx else 0.0
    return slope, my - slope * mx


def model_context(results_dir: Path, dt_of):
    """Return per-setter (n_switch, reconfig_us, model_time_us, energy_uj).

    Ground truth for the per-inference NPU dispatch sequence is measurements.csv
    (its distinct (layer, layer_idx) kernel rows); each dispatch's footprint
    (numCores) is looked up from configurations.json by GEMM-type name. This
    correctly excludes CPU-fallback GEMMs (e.g. attention_score/context) and
    counts split GEMMs (Q/K/V) individually.
    """
    import json
    json_path = results_dir / "measurements.json"
    csv_path = results_dir / "measurements.csv"
    cfg_path = results_dir / "configurations.json"
    if not (json_path.exists() and csv_path.exists() and cfg_path.exists()):
        return None

    conf_list = json.loads(cfg_path.read_text())
    conf_list = conf_list.get("configurations", conf_list)
    # (setter, gemm_type_name) -> numCores
    type_cores = {}
    for c in conf_list:
        if "error" in c:
            continue
        type_cores[(c.get("setter"), c["shape"]["layer"])] = \
            c["config"]["tc_entry"]["numCores"]

    def lookup_cores(setter, layer_str):
        # longest configured GEMM-type name that is a substring of the
        # measured layer key (e.g. "attention_qkv" in "L0/attention_qkv/Q").
        best, best_len = None, -1
        for (s, name), nc in type_cores.items():
            if s == setter and name in layer_str and len(name) > best_len:
                best, best_len = nc, len(name)
        return best

    # Distinct per-inference NPU dispatches per setter.
    inst = defaultdict(set)
    for r in csv.DictReader(csv_path.open()):
        if r.get("measurement_type") == "kernel" and \
                r.get("layer") not in ("", "_"):
            inst[r["setter"]].add((r["layer"], r.get("layer_idx")))

    doc = json.loads(json_path.read_text())
    out = {}
    for run in doc.get("runs", []):
        if run.get("backend") not in ("npu", "xrt"):
            continue
        s = run.get("setter", "?")
        cores = [lookup_cores(s, lyr) for (lyr, _idx) in inst.get(s, set())]
        cores = [c for c in cores if c is not None]
        out[s] = (len(cores),
                  sum(dt_of(c) for c in cores),
                  run.get("batch_min_model_time_us", -1.0),
                  run.get("batch_min_energy_per_inference_uj", -1.0))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path,
                   default=ONNX_DIR / "out" / "reports" / "reconfig_v1.csv")
    p.add_argument("--run-tag", default=None,
                   help="Which run_tag to analyze (default: latest).")
    p.add_argument("--fig-dir", type=Path,
                   default=ONNX_DIR / "paper" / "fig" / "data")
    args = p.parse_args(argv)

    rows = load_reconfig(args.csv, args.run_tag)
    if not rows:
        print(f"no rows in {args.csv}")
        return 1
    deltas = compute_deltas(rows)

    print("== per-switch reconfiguration cost ==")
    print(f"{'shape':28s} {'d':>2s} {'base_us':>9s} {'alt_us':>9s} "
          f"{'dt_us':>8s} {'base_uJ':>9s} {'alt_uJ':>9s} {'dE_uJ':>8s}")
    for (shape, d), v in deltas.items():
        def f(x, w=9, p=2):
            return (f"%{w}.{p}f" % x) if x is not None else " " * (w - 1) + "-"
        print(f"{shape:28s} {d:2d} {f(v['base_t'])} {f(v['alt_t'])} "
              f"{f(v['dt'],8)} {f(v['base_e'])} {f(v['alt_e'])} {f(v['de'],8)}")

    # Figure data: per shape/depth deltas.
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_csv = args.fig_dir / "reconfig_v1.csv"
    with fig_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["shape_tag", "depth", "base_us", "alt_us", "dt_us",
                    "base_uj", "alt_uj", "de_uj"])
        for (shape, d), v in deltas.items():
            w.writerow([shape, d, v["base_t"], v["alt_t"], v["dt"],
                        v["base_e"], v["alt_e"], v["de"]])
    print(f"\nwrote {fig_csv}")

    # Build numCores -> Delta_t map from measured rows + linear fit for
    # interpolation of unmeasured footprints (e.g. MLP fc2 = 8 cores).
    core_dt = {}
    for r in rows:
        if r["schedule"] == "alternating":
            key = (r["shape_tag"], int(r["depth"]))
            d = deltas.get(key, {}).get("dt")
            if d is not None:
                core_dt[int(r["numCores"])] = d
    if len(core_dt) >= 2:
        slope, icpt = linear_fit(sorted(core_dt.items()))
    else:
        slope, icpt = (0.0, 0.0)

    def dt_of(nc):
        if nc in core_dt:
            return core_dt[nc]
        return slope * nc + icpt  # linear interpolation

    print(f"\nper-switch Delta_t by footprint (measured): "
          f"{ {k: round(v,1) for k,v in sorted(core_dt.items())} }")
    print(f"linear fit: Dt = {slope:.2f}*cores + {icpt:.1f} us/core "
          f"(for interpolation)")

    # Model contextualization (footprint-weighted, per actual STAR-Map picks).
    print("\n== model contextualization (footprint-weighted) ==")
    ctx_rows = []
    for tag, sub in (("mlp", "mlp_512_512_bs32"),
                     ("bert_small", "bert_small_L128_bs1")):
        mc = model_context(EXPERIMENTS / "results" / sub, dt_of)
        if not mc:
            print(f"  {tag}: no measurements/configurations found, skipping")
            continue
        for setter, (n_sw, rt, mt, me) in sorted(mc.items()):
            rt_pct = 100.0 * rt / mt if mt > 0 else -1
            print(f"  [{tag}] {setter:12s} n_switch={n_sw:3d} "
                  f"reconfig={rt/1000:6.2f} ms  model={mt/1000:8.2f} ms  "
                  f"-> time {rt_pct:5.2f}%")
            ctx_rows.append([tag, setter, n_sw, rt, mt, me, rt_pct])

    ctx_csv = args.fig_dir / "reconfig_model_context.csv"
    with ctx_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "setter", "n_switch", "reconfig_time_us",
                    "model_time_us", "model_energy_uj", "time_pct"])
        w.writerows(ctx_rows)
    print(f"\nwrote {ctx_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

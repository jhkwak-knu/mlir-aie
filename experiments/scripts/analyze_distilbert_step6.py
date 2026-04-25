"""Step 6 (Notion #22) — DistilBERT setter-by-setter analysis.

Produces:
  * <results_dir>/analysis_step6/setter_configs.csv   — 24-row tile-param table
  * <results_dir>/analysis_step6/edp_decomposition.csv — per-gemm_type kernel EDP
  * <results_dir>/analysis_step6/per_setter_summary.csv — model-level T / E / EDP
  * <results_dir>/analysis_step6/setter_configs.md    — markdown tables + commentary

Usage:
  python experiments/scripts/analyze_distilbert_step6.py \\
      --results-dir experiments/results/distilbert_L128_bs1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _df_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Pandas-free markdown table writer (tabulate dep optional).

    Robust to pivoted frames whose index has a name (we promote it to a
    column) and to NaN floats (rendered as empty cells without invoking
    pd.isna on Series).
    """
    if df.index.name is not None or any(n is not None for n in df.index.names):
        df = df.reset_index()
    else:
        df = df.copy()

    def fmt(v):
        try:
            if v is None:
                return ""
            if isinstance(v, float):
                if v != v:  # NaN
                    return ""
                return f"{v:{floatfmt}}"
            if isinstance(v, (list, tuple)):
                return str(v)
            return str(v)
        except Exception:
            return str(v)

    # Collapse duplicate column names so itertuples-style access is unique.
    df.columns = [str(c) for c in df.columns]
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            # If duplicate columns exist, r[c] may be a Series. Pick first.
            if hasattr(v, "iloc"):
                v = v.iloc[0]
            cells.append(fmt(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


SHAPE_ORDER = [
    "attention_qkv", "attention_output", "attention_score",
    "attention_context", "ffn_expand", "ffn_compress",
]
SETTER_ORDER = ["star_map", "max_p", "timeloop", "charm_cdse"]
NPU_DISPATCHED = {"attention_qkv", "attention_output", "ffn_expand", "ffn_compress"}


def load_configs(results_dir: Path) -> pd.DataFrame:
    """24-row table: every (setter, shape) tile config."""
    path = results_dir / "configurations.json"
    cfg = json.loads(path.read_text())
    rows = []
    for c in cfg.get("configurations", []):
        s = c.get("shape", {}) or {}
        cfg_ = c.get("config", {}) or {}
        tc = cfg_.get("tc_entry", {}) or {}
        lev = (tc.get("levels") or [{}])[0]
        rows.append({
            "shape_label": s.get("layer"),
            "M": s.get("M"), "K": s.get("K"), "N": s.get("N"),
            "setter": c.get("setter"),
            "P_cores": cfg_.get("P"),
            "SPm": lev.get("SPm"), "SPn": lev.get("SPn"),
            "TPm": lev.get("TPm"), "TPk": lev.get("TPk"), "TPn": lev.get("TPn"),
            "TM": lev.get("TM"), "TK": lev.get("TK"), "TN": lev.get("TN"),
            "tpOrder": str(lev.get("tpOrder")),
            "doubleBuffer": tc.get("doubleBuffer"),
            "t_total_pred_us": tc.get("t_total_pred"),
            "npu_dispatched": s.get("layer") in NPU_DISPATCHED,
        })
    df = pd.DataFrame(rows)
    df["shape_rank"] = df["shape_label"].map({s: i for i, s in enumerate(SHAPE_ORDER)})
    df["setter_rank"] = df["setter"].map({s: i for i, s in enumerate(SETTER_ORDER)})
    return df.sort_values(["shape_rank", "setter_rank"]).drop(
        columns=["shape_rank", "setter_rank"]).reset_index(drop=True)


def load_measurements(results_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(results_dir / "measurements.csv")
    df["e_per_inf_uj"] = df.apply(
        lambda r: (r["energy_uj_min_package"] / r["n_inner"]) if r["n_inner"] else 0.0,
        axis=1,
    )
    return df


def aggregate_per_setter_kernel(df: pd.DataFrame) -> pd.DataFrame:
    """Min-of-batch per (setter, layer) for kernel rows."""
    k = df[df["measurement_type"] == "kernel"]
    g = k.groupby(["setter", "layer", "gemm_type", "layer_idx", "sub_type"],
                  as_index=False, dropna=False).agg(
        t_min=("time_us_min", "min"),
        e_min=("e_per_inf_uj", "min"),
    )
    g["edp"] = g["t_min"] * g["e_min"]
    return g


def edp_decomposition_by_gemm_type(kernel_agg: pd.DataFrame) -> pd.DataFrame:
    """Sum per (setter, gemm_type) — matches what one inference accumulates."""
    g = kernel_agg.groupby(["setter", "gemm_type"], as_index=False).agg(
        t_us_sum=("t_min", "sum"),
        e_uj_sum=("e_min", "sum"),
    )
    g["edp"] = g["t_us_sum"] * g["e_uj_sum"]
    g["avg_pwr_mW"] = g["e_uj_sum"] / g["t_us_sum"] * 1000  # uJ/us = W; *1000 = mW
    return g


def per_setter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Model-level (per-inference, min across batches) for each setter."""
    m = df[df["measurement_type"] == "model"]
    g = m.groupby("setter", as_index=False).agg(
        t_us_min=("time_us_min", "min"),
        e_uj_min=("e_per_inf_uj", "min"),
    )
    g["edp"] = g["t_us_min"] * g["e_uj_min"]
    g["t_ms"] = g["t_us_min"] / 1000
    g["e_mJ"] = g["e_uj_min"] / 1000
    g["edp_M"] = g["edp"] / 1e6
    g["edp_rel"] = g["edp"] / g["edp"].min()
    return g.sort_values("edp").reset_index(drop=True)


def write_markdown_report(out_md: Path, configs: pd.DataFrame,
                          gemm_decomp: pd.DataFrame,
                          summary: pd.DataFrame) -> None:
    lines = []
    lines.append("# DistilBERT Step 6 — Setter Configuration & EDP Analysis\n")
    lines.append("Generated by `experiments/scripts/analyze_distilbert_step6.py`.\n")

    # Section 1: per-setter model-level summary
    lines.append("## 1. Model-level (per-inference, min across 5 batches)\n")
    cols = ["setter", "t_ms", "e_mJ", "edp_M", "edp_rel"]
    lines.append(summary[cols].pipe(lambda d: _df_to_markdown(d, ".3f")))
    lines.append("")

    # Section 2: EDP by gemm_type
    lines.append("\n## 2. Kernel-level EDP per gemm_type (sum of NPU dispatches)\n")
    piv_t = gemm_decomp.pivot(index="gemm_type", columns="setter", values="t_us_sum")
    piv_e = gemm_decomp.pivot(index="gemm_type", columns="setter", values="e_uj_sum")
    piv_edp = gemm_decomp.pivot(index="gemm_type", columns="setter", values="edp")
    piv_edp_rel = piv_edp.divide(piv_edp.min(axis=1), axis=0)

    lines.append("### 2a. time_us_sum per gemm_type\n")
    lines.append(piv_t.round(0).pipe(lambda d: _df_to_markdown(d, ".0f")))
    lines.append("\n### 2b. energy_uj_sum per gemm_type\n")
    lines.append(piv_e.round(0).pipe(lambda d: _df_to_markdown(d, ".0f")))
    lines.append("\n### 2c. EDP normalized to row min\n")
    lines.append(piv_edp_rel.round(4).pipe(lambda d: _df_to_markdown(d, ".4f")))
    lines.append("")

    lines.append("\n### 2d. Winner per gemm_type\n")
    for gtype in piv_edp.index:
        row = piv_edp.loc[gtype]
        ranking = row.sort_values()
        ranks = " > ".join(
            f"{s} ({row[s]/ranking.iloc[0]:.3f})" for s in ranking.index
        )
        lines.append(f"- **{gtype}**: {ranks}")
    lines.append("")

    # Section 3: tile params
    lines.append("\n## 3. Tile parameters per (setter, shape)\n")
    cols = ["shape_label", "M", "K", "N", "setter", "P_cores",
            "SPm", "SPn", "TPm", "TPk", "TPn", "TM", "TK", "TN",
            "tpOrder", "npu_dispatched"]
    lines.append(configs[cols].pipe(lambda d: _df_to_markdown(d)))
    lines.append("")

    # Section 4: P (cores) summary
    lines.append("\n## 4. Did every setter pick P=32?\n")
    p_used = configs.groupby("setter")["P_cores"].apply(
        lambda s: sorted(set(s))).rename("P_values_used")
    lines.append(p_used.to_frame().pipe(lambda d: _df_to_markdown(d)))
    lines.append("")
    p_per_shape = configs.pivot(index="shape_label", columns="setter",
                                values="P_cores")
    p_per_shape = p_per_shape.reindex(SHAPE_ORDER)
    lines.append("\n### P per (shape, setter)\n")
    lines.append(p_per_shape.pipe(lambda d: _df_to_markdown(d, ".0f")))
    lines.append("")

    # Section 5: distinct config sigs
    def _sig(r):
        return (f"P={r.P_cores} SP={r.SPm}/{r.SPn} TP={r.TPm}/{r.TPk}/{r.TPn} "
                f"T={r.TM}/{r.TK}/{r.TN} ord={r.tpOrder}")
    configs["sig"] = configs.apply(_sig, axis=1)
    distinct = configs.groupby(["shape_label", "M", "K", "N"])["sig"].nunique()
    distinct = distinct.rename("distinct_configs_across_setters").reset_index()
    distinct["shape_rank"] = distinct["shape_label"].map(
        {s: i for i, s in enumerate(SHAPE_ORDER)})
    distinct = distinct.sort_values("shape_rank").drop(columns=["shape_rank"])
    lines.append("\n## 5. Distinct configs per shape (across the 4 setters)\n")
    lines.append(distinct.pipe(lambda d: _df_to_markdown(d)))
    lines.append("")

    # Section 6: STAR-Map ≡ max_p
    lines.append("\n## 6. STAR-Map vs max_p — identical configs?\n")
    sm = configs[configs["setter"] == "star_map"][
        ["shape_label", "M", "K", "N", "sig"]].rename(columns={"sig": "star_sig"})
    mp = configs[configs["setter"] == "max_p"][
        ["shape_label", "M", "K", "N", "sig"]].rename(columns={"sig": "max_p_sig"})
    cmp_ = sm.merge(mp, on=["shape_label", "M", "K", "N"])
    cmp_["identical"] = cmp_["star_sig"] == cmp_["max_p_sig"]
    cmp_["shape_rank"] = cmp_["shape_label"].map(
        {s: i for i, s in enumerate(SHAPE_ORDER)})
    cmp_ = cmp_.sort_values("shape_rank").drop(columns=["shape_rank"])
    lines.append(cmp_[["shape_label", "M", "K", "N", "identical"]]
                 .pipe(lambda d: _df_to_markdown(d)))
    lines.append("")

    # Key finding (derived from the data, not hardcoded)
    lines.append("\n## 7. Key finding (data-derived)\n")

    # Model-level ranking
    summary_sorted = summary.sort_values("edp_rel")
    rank = list(summary_sorted["setter"])
    best = summary_sorted.iloc[0]
    rank_str = " < ".join(f"{r['setter']} ({r['edp_rel']:.3f})"
                          for _, r in summary_sorted.iterrows())
    lines.append(f"- Model-level EDP ranking (lower better): {rank_str}.")

    # Best vs second
    if len(summary_sorted) >= 2:
        second = summary_sorted.iloc[1]
        # Signed deltas: positive = runner-up is worse (winner saves on that
        # metric); negative = runner-up beats winner on this metric (winner's
        # EDP lead came from the *other* axis).
        delta_t_pct = (second["t_us_min"] - best["t_us_min"]) / best["t_us_min"] * 100
        delta_e_pct = (second["e_uj_min"] - best["e_uj_min"]) / best["e_uj_min"] * 100
        delta_edp_pct = (second["edp"] - best["edp"]) / best["edp"] * 100
        lines.append(
            f"- **{best['setter']}** wins (lower EDP). vs runner-up "
            f"`{second['setter']}` (signed gap, +ve = winner ahead): "
            f"time {delta_t_pct:+.1f}%, energy {delta_e_pct:+.1f}%, "
            f"EDP {delta_edp_pct:+.1f}%."
        )

    # NPU-dispatched shapes where star_map and max_p differ
    sm_mp = cmp_[["shape_label", "identical"]]
    npu_shapes = set(configs[configs["npu_dispatched"]]["shape_label"])
    differ_npu = [r["shape_label"] for _, r in sm_mp.iterrows()
                  if not r["identical"] and r["shape_label"] in npu_shapes]
    same_npu = [r["shape_label"] for _, r in sm_mp.iterrows()
                if r["identical"] and r["shape_label"] in npu_shapes]
    lines.append(
        f"- NPU-dispatched shapes where star_map != max_p: "
        f"{len(differ_npu)} ({', '.join(differ_npu) or '-'})."
    )
    lines.append(
        f"- NPU-dispatched shapes where star_map == max_p: "
        f"{len(same_npu)} ({', '.join(same_npu) or '-'})."
    )

    # P-core picks per setter
    p_per_setter = (configs.groupby("setter")["P_cores"]
                    .agg(lambda s: sorted(set(int(x) for x in s)))
                    .to_dict())
    p_str = "; ".join(f"{k}: {v}" for k, v in p_per_setter.items())
    lines.append(f"- P-cores picked per setter (across all 6 shapes): {p_str}.")

    # Per-gemm_type winners
    gemm_winners: List[str] = []
    for gemm in sorted(gemm_decomp["gemm_type"].unique()):
        sub = gemm_decomp[gemm_decomp["gemm_type"] == gemm].sort_values("edp")
        w = sub.iloc[0]["setter"]
        gemm_winners.append(f"{gemm}->{w}")
    lines.append(f"- Per-gemm_type EDP winners: {', '.join(gemm_winners)}.")

    lines.append("")

    out_md.write_text("\n".join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path("experiments/results/distilbert_L128_bs1"),
                        help="dir holding configurations.json + measurements.csv")
    args = parser.parse_args(argv)

    rd = args.results_dir
    out_dir = rd / "analysis_step6"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = load_configs(rd)
    df = load_measurements(rd)
    kagg = aggregate_per_setter_kernel(df)
    decomp = edp_decomposition_by_gemm_type(kagg)
    summary = per_setter_summary(df)

    configs.to_csv(out_dir / "setter_configs.csv", index=False)
    decomp.to_csv(out_dir / "edp_decomposition.csv", index=False)
    summary.to_csv(out_dir / "per_setter_summary.csv", index=False)
    write_markdown_report(out_dir / "setter_configs.md", configs, decomp, summary)

    print(f"wrote {out_dir}/setter_configs.csv")
    print(f"wrote {out_dir}/edp_decomposition.csv")
    print(f"wrote {out_dir}/per_setter_summary.csv")
    print(f"wrote {out_dir}/setter_configs.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

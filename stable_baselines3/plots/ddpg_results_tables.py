"""
DDPG / SUFT (DDPGSL) results: score tables + analysis.

For every *completed* run the final evaluation score (last ``Eval num_timesteps=..., episode_reward=...``
line) is taken, averaged over runs per configuration and environment, and every SUFT configuration is
compared against the baseline DDPG configuration with the same timesteps / batch size / train freq
(and any other non-SUFT header key):

    ratio = suft_mean_score / baseline_mean_score * 100%   (100% = same as baseline)

A run counts as completed when its log ends with ``Total train runtime`` and its last evaluation is at
the target number of timesteps.

Outputs (in --out-dir):
    runs.csv              one row per log file (config, env, run, completed, final score)
    summary.csv           one row per configuration x env (n, mean, std, baseline, ratio, p-value)
    config_summary.csv    one row per SUFT configuration (aggregated over envs)
    ddpg_results.md       all tables + analysis
    figures/*.png         heatmaps / bar charts of the ratios

Usage:
    python stable_baselines3/plots/ddpg_results_tables.py
    python stable_baselines3/plots/ddpg_results_tables.py --results-dir <dir> --out-dir <dir>
"""

import argparse
import math
import os
import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd

DEFAULT_RESULTS_DIR = r"C:\Users\sorte\Desktop\BIU\phd\suft2\ICML_2027\results"
ALGO_DIRS = ("Baseline/DDPG", "SUFT/DDPG", "SUFT/DDPGSL")
ENV_ORDER = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5", "Walker2d-v5", "Swimmer-v5"]

# DDPG / DDPGSL defaults before commit 46df27e. Logs from that time do not print batch_size / train_freq.
LEGACY_BATCH, LEGACY_TRAIN_FREQ = 256, 1

# Header keys that define the algorithm / SUFT variant; every other header key must match the baseline.
KNOWN_KEYS = {
    "device", "tf_lambda", "agent_name", "timesteps", "run_number", "gym_env_name",
    "buffer_size", "batch_size", "train_freq", "short_buffer_size", "sampling_fraction",
}
CFG_COLS = ["algo_dir", "agent", "lam", "short_buffer", "fraction", "timesteps", "batch", "train_freq", "legacy", "extra"]
MATCH_COLS = ["timesteps", "batch", "train_freq", "extra"]  # baseline <-> SUFT matching key (+ env)
FACTORS = {"lam": "λ", "short_buffer": "Short", "fraction": "Frac"}

FILE_RE = re.compile(r"^(?P<env>[A-Za-z0-9]+-v\d+)_(?P<cfg>.+)_run_(?P<run>\d+)\.log$")
EVAL_RE = re.compile(r"Eval num_timesteps=(\d+), episode_reward=([-+\d.eE]+|nan)")
TAIL_BYTES = 16_000
ALPHA = 0.05
COLLAPSE_RATIO = 50.0  # a configuration whose mean ratio is below this is counted as "collapsed"


# ----------------------------------------------------------------------------------------------------------
# Statistics (Welch t-test without scipy)
# ----------------------------------------------------------------------------------------------------------
def _betacf(a, b, x, max_iter=500, eps=3e-14):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_bt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_bt) * _betacf(b, a, 1.0 - x) / b


def welch_pvalue(x, y):
    """Two-sided Welch t-test p-value for mean(x) != mean(y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se2 = vx + vy
    if se2 == 0:
        return np.nan
    t = (x.mean() - y.mean()) / math.sqrt(se2)
    df = se2**2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    return betainc(df / 2.0, 0.5, df / (df + t * t))


# ----------------------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------------------
def parse_log(path):
    with open(path, "rb") as fh:
        head = fh.read(4000).decode("utf8", "ignore")
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - TAIL_BYTES))
        tail = fh.read().decode("utf8", "ignore")

    header = {}
    for line in head.splitlines():
        if line.startswith("Using "):
            break
        parts = line.strip().split(" ", 1)
        if len(parts) == 2 and re.fullmatch(r"[a-z_]+", parts[0]):
            header[parts[0]] = parts[1].strip()

    evals = EVAL_RE.findall(tail)
    if not evals:  # killed early: the last evaluation is not in the tail
        with open(path, encoding="utf8", errors="ignore") as fh:
            evals = EVAL_RE.findall(fh.read())
    last_t, last_score = (int(evals[-1][0]), float(evals[-1][1])) if evals else (0, np.nan)

    timesteps = int(float(header.get("timesteps", "0")))
    completed = "Total train runtime" in tail and last_t >= timesteps > 0
    extra = ";".join(f"{k}={v}" for k, v in sorted(header.items()) if k not in KNOWN_KEYS)
    return dict(
        env=header.get("gym_env_name"),
        agent=header.get("agent_name", "?"),
        lam=float(header.get("tf_lambda", 0)),
        short_buffer=int(header.get("short_buffer_size", 0)),
        fraction=float(header.get("sampling_fraction", 0)),
        timesteps=timesteps,
        batch=int(header.get("batch_size", LEGACY_BATCH)),
        train_freq=int(header.get("train_freq", LEGACY_TRAIN_FREQ)),
        legacy="batch_size" not in header,
        extra=extra,
        n_evals=len(evals),
        last_eval_t=last_t,
        last_eval_score=last_score,
        completed=completed,
        final_score=last_score if completed else np.nan,
    )


def load_runs(results_dir, algo_dirs):
    recs = []
    for sub in algo_dirs:
        d = os.path.join(results_dir, *sub.split("/"))
        if not os.path.isdir(d):
            print(f"[warn] missing directory: {d}")
            continue
        files = sorted(f for f in os.listdir(d) if FILE_RE.match(f))
        print(f"parsing {len(files):5d} logs in {sub}")
        for f in files:
            m = FILE_RE.match(f)
            rec = parse_log(os.path.join(d, f))
            rec.update(algo_dir=sub, file=f, run=int(m["run"]))
            rec["env"] = rec["env"] or m["env"]
            recs.append(rec)
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------------------------------------
def is_baseline(algo_dir):
    return algo_dir.startswith("Baseline")


def summarize(runs):
    done = runs[runs.completed]
    g = done.groupby(CFG_COLS + ["env"], sort=False).final_score
    summ = g.agg(n="count", mean="mean", std="std").reset_index()
    summ["scores"] = g.apply(list).values
    summ["sem"] = summ["std"] / np.sqrt(summ["n"])
    summ["is_baseline"] = summ.algo_dir.map(is_baseline)

    base = {}
    for _, r in summ[summ.is_baseline].iterrows():
        base[tuple(r[c] for c in MATCH_COLS) + (r.env, r.legacy)] = r

    out_cols = {k: [] for k in ("base_mean", "base_n", "ratio", "pvalue", "alt_base_mean", "alt_ratio", "base_legacy")}
    for _, r in summ.iterrows():
        key = tuple(r[c] for c in MATCH_COLS) + (r.env,)
        prim = None if r.is_baseline else base.get(key + (r.legacy,)) if key + (r.legacy,) in base else base.get(key + (not r.legacy,))
        alt = None if r.is_baseline or prim is None else base.get(key + (not prim.legacy,))
        out_cols["base_mean"].append(prim["mean"] if prim is not None else np.nan)
        out_cols["base_n"].append(prim["n"] if prim is not None else np.nan)
        out_cols["base_legacy"].append(prim["legacy"] if prim is not None else np.nan)
        out_cols["ratio"].append(100 * r["mean"] / prim["mean"] if prim is not None else np.nan)
        out_cols["pvalue"].append(welch_pvalue(r.scores, prim.scores) if prim is not None else np.nan)
        out_cols["alt_base_mean"].append(alt["mean"] if alt is not None else np.nan)
        out_cols["alt_ratio"].append(100 * r["mean"] / alt["mean"] if alt is not None else np.nan)
    for k, v in out_cols.items():
        summ[k] = v
    summ["sig"] = np.where(summ.pvalue < ALPHA, np.sign(summ.ratio - 100), 0)
    return summ


def config_summary(summ):
    s = summ[~summ.is_baseline & summ.ratio.notna()]
    agg = s.groupby(CFG_COLS, sort=False).agg(
        n_envs=("env", "count"),
        mean_ratio=("ratio", "mean"),
        median_ratio=("ratio", "median"),
        min_ratio=("ratio", "min"),
        alt_mean_ratio=("alt_ratio", "mean"),
        n_better=("ratio", lambda x: int((x > 100).sum())),
        n_sig_better=("sig", lambda x: int((x > 0).sum())),
        n_sig_worse=("sig", lambda x: int((x < 0).sum())),
    )
    return agg.reset_index()


# ----------------------------------------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------------------------------------
def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.1f}"


def pct(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0f}%"


def env_short(e):
    return e.split("-")[0]


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and math.isnan(v)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def ordered_envs(envs):
    envs = set(envs)
    return [e for e in ENV_ORDER if e in envs] + sorted(envs - set(ENV_ORDER))


def group_key(r):
    return (r.algo_dir, r.timesteps, r.batch, r.train_freq, r.legacy, r.extra)


def group_title(key):
    algo_dir, T, b, f, legacy, extra = key
    t = f"{algo_dir}: {T / 1e6:g}M steps, batch {b}, train freq {f}"
    if extra:
        t += f", {extra}"
    if legacy:
        t += " (legacy logs: batch/train freq = old defaults)"
    return t


def varying_factors(df):
    return [c for c in FACTORS if df[c].nunique() > 1]


# ----------------------------------------------------------------------------------------------------------
# Report sections
# ----------------------------------------------------------------------------------------------------------
def section_status(runs):
    st = runs.groupby(CFG_COLS, sort=False).agg(total=("completed", "size"), completed=("completed", "sum"),
                                               envs=("env", "nunique")).reset_index()
    inc = runs[~runs.completed]
    L = ["## 1. Run status", "",
         f"{len(runs)} log files, **{int(runs.completed.sum())} completed**, {len(inc)} incomplete, "
         f"{st.shape[0]} configurations ({int((st.completed == 0).sum())} with no completed run).", ""]
    if len(inc):
        g = inc.groupby(CFG_COLS + ["env"], sort=False).agg(
            incomplete=("run", "count"), max_step=("last_eval_t", "max"), min_step=("last_eval_t", "min")).reset_index()
        tot = runs.groupby(CFG_COLS + ["env"], sort=False).run.count().rename("runs").reset_index()
        g = g.merge(tot, on=CFG_COLS + ["env"])
        L += ["Incomplete runs (excluded):", ""]
        L.append(md_table(pd.DataFrame({
            "Folder": g.algo_dir, "Agent": g.agent, "λ": g.lam.map("{:g}".format),
            "Short": g.short_buffer.map(lambda v: v or "–"), "Steps": g.timesteps.map(lambda v: f"{v / 1e6:g}M"),
            "Batch": g.batch, "TF": g.train_freq, "Env": g.env.map(env_short),
            "Incomplete / runs": [f"{a}/{b}" for a, b in zip(g.incomplete, g.runs)],
            "Reached": [f"{a / 1e6:g}M–{b / 1e6:g}M" for a, b in zip(g.min_step, g.max_step)],
        })))
        L.append("")
    return L


def section_baselines(summ):
    b = summ[summ.is_baseline]
    envs = ordered_envs(b.env)
    L = ["## 2. Baseline DDPG (final eval, mean ± std, n = completed runs)", ""]
    rows = []
    for key, g in b.groupby(["timesteps", "batch", "train_freq", "legacy", "extra"], sort=True):
        T, bs, tf, legacy, extra = key
        cells = {e: f"{fmt(r['mean'])} ± {fmt(r['std'])} (n={r['n']})" for _, r in g.iterrows() for e in [r.env]}
        rows.append({"Steps": f"{T / 1e6:g}M", "Batch": bs, "TF": tf,
                     "Note": ("legacy" if legacy else "") + (f" {extra}" if extra else ""),
                     **{env_short(e): cells.get(e, "–") for e in envs}})
    L += [md_table(pd.DataFrame(rows)), ""]
    return L


def factor_label_cols(df):
    out = {}
    for c, name in FACTORS.items():
        if c == "lam" or df[c].nunique() > 1:
            out[name] = df[c].map(lambda v: f"{v:g}")
    return out


def section_group(key, gsum, gcfg, idx):
    envs = ordered_envs(gsum.env)
    b_legacy = gsum.base_legacy.dropna().unique()
    has_alt = gsum.alt_ratio.notna().any()
    base_rows = gsum.drop_duplicates("env").set_index("env")
    L = [f"### 3.{idx} {group_title(key)}", ""]
    L.append("Baseline" + (" (legacy)" if len(b_legacy) and b_legacy[0] else "") + ": " +
             ", ".join(f"{env_short(e)} {fmt(base_rows.loc[e, 'base_mean'])} (n={int(base_rows.loc[e, 'base_n'])})"
                       for e in envs if not math.isnan(base_rows.loc[e, "base_mean"])))
    L += ["", "Cell = mean final score (ratio vs baseline). ▲/▼ = significantly better/worse (Welch t-test, "
              f"p < {ALPHA}).", ""]
    gcfg = gcfg.sort_values(list(FACTORS)).reset_index(drop=True)
    table = pd.DataFrame(factor_label_cols(gcfg))
    table["Runs"] = [int(gsum[(gsum[CFG_COLS] == r[CFG_COLS]).all(axis=1)].n.max()) for _, r in gcfg.iterrows()]
    for e in envs:
        col = []
        for _, r in gcfg.iterrows():
            m = gsum[(gsum[CFG_COLS] == r[CFG_COLS]).all(axis=1) & (gsum.env == e)]
            if m.empty:
                col.append("–")
                continue
            m = m.iloc[0]
            mark = {1: " ▲", -1: " ▼"}.get(int(m.sig), "")
            col.append(f"{fmt(m['mean'])} ({pct(m.ratio)}){mark}")
        table[env_short(e)] = col
    table["Mean ratio"] = gcfg.mean_ratio.map(lambda v: f"**{pct(v)}**")
    table["Median ratio"] = gcfg.median_ratio.map(pct)
    table["Better / sig+ / sig−"] = [f"{a}/{n} · {b} · {c}" for a, b, c, n in
                                    zip(gcfg.n_better, gcfg.n_sig_better, gcfg.n_sig_worse, gcfg.n_envs)]
    if has_alt:
        table["Mean ratio vs other baseline"] = gcfg.alt_mean_ratio.map(pct)
    L += [md_table(table), ""]
    if has_alt:
        L += ["*Other baseline* = the baseline with the same hyper-parameters from the other code era "
              "(legacy vs. explicit batch/train-freq logs).", ""]
    return L


def section_factors(key, gcfg, idx):
    fac = varying_factors(gcfg)
    if not fac:
        return []
    L = [f"#### Factor effects – {group_title(key)}", "",
         "Mean (over configurations) of the per-configuration mean ratio; "
         f"'collapsed' = share of configurations with mean ratio < {COLLAPSE_RATIO:.0f}%.", ""]
    for c in fac:
        g = gcfg.groupby(c).agg(configs=("mean_ratio", "size"), mean_ratio=("mean_ratio", "mean"),
                                median_ratio=("median_ratio", "median"),
                                collapsed=("mean_ratio", lambda x: (x < COLLAPSE_RATIO).mean())).reset_index()
        L += [md_table(pd.DataFrame({FACTORS[c]: g[c].map("{:g}".format), "Configs": g.configs,
                                     "Mean ratio": g.mean_ratio.map(pct), "Median of median ratios": g.median_ratio.map(pct),
                                     "Collapsed": g.collapsed.map(lambda v: f"{100 * v:.0f}%")})), ""]
    for a, b in combinations(fac, 2):
        p = gcfg.pivot_table(index=a, columns=b, values="mean_ratio", aggfunc="mean")
        p.index = [f"{FACTORS[a]}={v:g}" for v in p.index]
        p.columns = [f"{FACTORS[b]}={v:g}" for v in p.columns]
        p = p.map(pct) if hasattr(p, "map") else p.applymap(pct)
        L += [f"Mean ratio, {FACTORS[a]} × {FACTORS[b]}:", "", md_table(p.reset_index().rename(columns={"index": ""})), ""]
    return L


def section_per_env(key, gsum):
    s = gsum[gsum.ratio.notna()]
    rows = []
    for e in ordered_envs(s.env):
        se = s[s.env == e]
        best = se.loc[se.ratio.idxmax()]
        lab = ", ".join(f"{FACTORS[c]}={best[c]:g}" for c in FACTORS if c == "lam" or se[c].nunique() > 1)
        rows.append({"Env": env_short(e), "Configs better than baseline": f"{(se.ratio > 100).sum()}/{len(se)}",
                     "Sig. better": int((se.sig > 0).sum()), "Sig. worse": int((se.sig < 0).sum()),
                     "Median ratio": pct(se.ratio.median()), "Best config": lab,
                     "Best score (ratio, p)": f"{fmt(best['mean'])} ({pct(best.ratio)}, p={best.pvalue:.3g})"})
    return [f"#### Per-environment view – {group_title(key)}", "", md_table(pd.DataFrame(rows)), ""]


def findings(groups):
    """Data-driven bullet points."""
    L = ["## 4. Key findings (auto-generated)", ""]
    for key, gsum, gcfg in groups:
        if gcfg.empty:
            continue
        L.append(f"**{group_title(key)}** – {len(gcfg)} configurations")
        best = gcfg.loc[gcfg.mean_ratio.idxmax()]
        worst = gcfg.loc[gcfg.mean_ratio.idxmin()]
        lab = lambda r: ", ".join(f"{FACTORS[c]}={r[c]:g}" for c in FACTORS if c == "lam" or gcfg[c].nunique() > 1)
        L.append(f"- Best: {lab(best)} → mean ratio {pct(best.mean_ratio)} (median {pct(best.median_ratio)}), "
                 f"better on {best.n_better}/{best.n_envs} envs, significantly better on {best.n_sig_better}, "
                 f"significantly worse on {best.n_sig_worse}.")
        robust = gcfg[(gcfg.n_sig_worse == 0) & (gcfg.mean_ratio > 100)].sort_values("mean_ratio", ascending=False)
        if len(robust):
            L.append("- Above baseline on average with no significantly-worse env: " +
                     "; ".join(f"{lab(r)} ({pct(r.mean_ratio)})" for _, r in robust.head(5).iterrows()) +
                     (f" (+{len(robust) - 5} more)" if len(robust) > 5 else "") + ".")
        else:
            L.append("- No configuration is above the baseline on average without a significantly-worse env.")
        L.append(f"- Worst: {lab(worst)} → mean ratio {pct(worst.mean_ratio)}.")
        n_coll = int((gcfg.mean_ratio < COLLAPSE_RATIO).sum())
        if n_coll:
            msgs = []
            for c in varying_factors(gcfg):
                rate = gcfg.groupby(c).mean_ratio.apply(lambda x: (x < COLLAPSE_RATIO).mean())
                bad = rate[rate >= 0.75]
                msgs += [f"{FACTORS[c]}={v:g} ({100 * r:.0f}% collapsed)" for v, r in bad.items()]
            L.append(f"- {n_coll}/{len(gcfg)} configurations collapse (mean ratio < {COLLAPSE_RATIO:.0f}%)" +
                     (": mostly " + ", ".join(msgs) if msgs else "") + ".")
        for c in varying_factors(gcfg):
            m = gcfg.groupby(c).mean_ratio.mean()
            L.append(f"- {FACTORS[c]}: best level {m.idxmax():g} ({pct(m.max())}), worst level {m.idxmin():g} "
                     f"({pct(m.min())}).")
        env_stats = gsum[gsum.ratio.notna()].groupby("env").agg(
            better=("ratio", lambda x: (x > 100).mean()), med=("ratio", "median"))
        hard = [env_short(e) for e, r in env_stats.iterrows() if r.better < 0.25]
        easy = [env_short(e) for e, r in env_stats.iterrows() if r.better > 0.6]
        if easy:
            L.append(f"- Envs where most configurations beat the baseline: {', '.join(easy)}.")
        if hard:
            L.append(f"- Envs where SUFT rarely beats the baseline (<25% of configurations): {', '.join(hard)}.")
        L.append("")
    return L


# ----------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------
def make_figures(groups, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    for i, (key, gsum, gcfg) in enumerate(groups, 1):
        if gcfg.empty:
            continue
        name = re.sub(r"[^A-Za-z0-9]+", "_", f"{key[0]}_{key[1] // 1_000_000}M_b{key[2]}_tf{key[3]}{'_legacy' if key[4] else ''}")
        fac = varying_factors(gcfg)
        if "lam" in fac and "short_buffer" in fac:
            facet = "fraction" if "fraction" in fac else None
            levels = sorted(gcfg[facet].unique()) if facet else [None]
            fig, axes = plt.subplots(1, len(levels), figsize=(3.4 * len(levels) + 1.2, 3.8), squeeze=False,
                                     layout="constrained")
            norm = TwoSlopeNorm(vmin=min(-100.0, gcfg.mean_ratio.min()), vcenter=100.0, vmax=max(200.0, gcfg.mean_ratio.max()))
            for k, (ax, lv) in enumerate(zip(axes[0], levels)):
                sub = gcfg if lv is None else gcfg[gcfg[facet] == lv]
                p = sub.pivot_table(index="lam", columns="short_buffer", values="mean_ratio")
                p = p.reindex(index=sorted(gcfg.lam.unique()), columns=sorted(gcfg.short_buffer.unique()))
                im = ax.imshow(p.values, cmap="RdBu", norm=norm, aspect="auto")
                ax.set_xticks(range(p.shape[1]), [f"{v:g}" for v in p.columns])
                ax.set_yticks(range(p.shape[0]), [f"{v:g}" for v in p.index] if k == 0 else [])
                ax.set_xlabel("short buffer")
                if k == 0:
                    ax.set_ylabel("λ")
                ax.set_title(f"fraction = {lv:g}" if lv is not None else "")
                for (y, x), v in np.ndenumerate(p.values):
                    ax.text(x, y, "" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=axes[0].tolist(), label="mean ratio vs baseline (%)")
            fig.suptitle(group_title(key), fontsize=9)
            path = os.path.join(fig_dir, f"heatmap_{name}.png")
            fig.savefig(path, dpi=130)
            plt.close(fig)
            paths.append(path)
        if len(gcfg) <= 12:  # per-env bars are readable only for small groups
            envs = ordered_envs(gsum.env)
            gcfg_s = gcfg.sort_values(list(FACTORS))
            width = 0.8 / max(1, len(gcfg_s))
            fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(envs)), 4))
            for j, (_, r) in enumerate(gcfg_s.iterrows()):
                vals = [gsum[(gsum[CFG_COLS] == r[CFG_COLS]).all(axis=1) & (gsum.env == e)].ratio.mean() for e in envs]
                lab = ", ".join(f"{FACTORS[c]}={r[c]:g}" for c in FACTORS if c == "lam" or gcfg[c].nunique() > 1)
                ax.bar(np.arange(len(envs)) + (j - (len(gcfg_s) - 1) / 2) * width, vals, width, label=lab)
            ax.axhline(100, color="k", lw=1, ls="--")
            ax.set_xticks(range(len(envs)), [env_short(e) for e in envs])
            ax.set_ylabel("score / baseline (%)")
            ax.set_title(group_title(key), fontsize=9)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            path = os.path.join(fig_dir, f"bars_{name}.png")
            fig.savefig(path, dpi=130)
            plt.close(fig)
            paths.append(path)
    return paths


# ----------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--algo-dirs", nargs="+", default=list(ALGO_DIRS), help="sub-folders of --results-dir to parse")
    ap.add_argument("--out-dir", default=None, help="default: <results-dir>/../analysis/DDPG")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.results_dir)), "analysis", "DDPG")
    os.makedirs(out_dir, exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    runs = load_runs(args.results_dir, args.algo_dirs)
    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    summ = summarize(runs)
    summ.drop(columns="scores").to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    cfg = config_summary(summ)
    cfg.to_csv(os.path.join(out_dir, "config_summary.csv"), index=False)

    suft = summ[~summ.is_baseline]
    groups = []
    for key in sorted({group_key(r) for r in suft.itertuples()}):
        gs = suft[[group_key(r) == key for r in suft.itertuples()]]
        gc = cfg[[group_key(r) == key for r in cfg.itertuples()]]
        groups.append((key, gs, gc))

    L = ["# DDPG vs. SUFT (DDPGSL) – final evaluation scores", "",
         "Score = last evaluation (`Eval num_timesteps=...`) of each completed run, averaged over runs. "
         "Ratio = SUFT mean / matched baseline mean × 100% (baseline = same timesteps, batch size, train freq; "
         "legacy logs without batch/train freq in the header use the old defaults "
         f"batch={LEGACY_BATCH}, train_freq={LEGACY_TRAIN_FREQ} and are matched to the legacy baseline first). "
         "*Mean ratio* averages the per-env ratios, so a negative score on one env (e.g. Ant) pulls it down "
         "strongly; *median ratio* is the robust alternative.", ""]
    L += section_status(runs)
    L += section_baselines(summ)
    L += ["## 3. SUFT configurations vs. baseline", ""]
    for i, (key, gs, gc) in enumerate(groups, 1):
        if gc.empty:
            continue
        L += section_group(key, gs, gc, i)
        L += section_per_env(key, gs)
        L += section_factors(key, gc, i)
    L += findings(groups)

    if not args.no_figures:
        figs = make_figures(groups, os.path.join(out_dir, "figures"))
        L += ["## 5. Figures", ""] + [f"![{os.path.basename(p)}](figures/{os.path.basename(p)})" for p in figs] + [""]

    md_path = os.path.join(out_dir, "ddpg_results.md")
    with open(md_path, "w", encoding="utf8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(findings(groups)))
    print(f"\nwrote {md_path}\n      {out_dir}{os.sep}runs.csv, summary.csv, config_summary.csv, figures/")


if __name__ == "__main__":
    main()

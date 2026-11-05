"""
Atari-57 results: baselines vs SUFT (short / long term buffer + SUFT term). BBF, DDPG, SAC and TD3 are excluded.

Every log of <results-dir>/Baseline/<agent> and <results-dir>/SUFT/<agent> (Atari agents only) is parsed:

    final eval score      last ``Eval num_timesteps=..., episode_reward=...`` line (evaluation at the last step)
    final training score  last ``rollout/ep_rew_mean`` of the logger tables (mean of the last 100 training episodes)
    eval curve            every ``Eval num_timesteps`` line
    training curve        every ``rollout/ep_rew_mean`` vs ``time/total_timesteps``

Both final scores are averaged over the runs per game, and every SUFT configuration is compared with its baseline
(``BASELINE_OF``, same number of timesteps):

    score ratio (%)   = 100 / |valid| * sum_{g in valid} (agent_g - random_g) / (baseline_g - random_g)
    improvement (%)   = score ratio - 100
    valid games       = games with agent_g - random_g > 1 and baseline_g - random_g > 1

Human normalized score  HNS = (score - random) / (human - random), aggregated as in rliable (Agarwal et al., 2021):

    mean / median     over the games of the per-game mean over runs
    IQM               interquartile mean of all the (run, game) scores (25% trimmed on each side)
    optimality gap    mean over all the (run, game) scores of max(0, 1 - HNS)
    95% CIs           stratified bootstrap over the runs of every game (--bootstrap samples)

Additional statistics per comparison: wins / ties / losses over games, exact two-sided sign test, Wilcoxon signed-rank
test on the per-game HNS differences (normal approximation), probability of improvement P(agent > baseline)
(rliable, with a bootstrap CI). An ablation compares every SUFT configuration with lambda > 0 with the same agent and
buffer at lambda = 0 (isolates the SUFT term from the short / long term buffer).

Outputs (in --out-dir, default <results-dir>/Atari_analysis):
    atari_results.md            every table + analysis
    comparison_table.tex        main comparison table (LaTeX, booktabs)
    configs.csv                 the configurations and their baseline
    runs.csv                    one row per log (config, game, run, final eval / training score)
    per_game.csv                one row per configuration x game x score (n, mean, std, sem, HNS)
    per_game_comparison.csv     one row per comparison x game x score (means, ratio, valid, HNS)
    comparison.csv              one row per comparison x score (ratio, improvement, tests, HNS metrics)
    aggregate.csv               one row per configuration x score (HNS mean / median / IQM / optimality gap + CIs)
    figures/games_<cfg>_<eval|train>.png      the 57 games, baseline vs configuration (mean +- s.e. over runs),
                                              random (gray) and human (red) scores as dashed lines
    figures/iqm_curves_<eval|train>.png       IQM HNS vs steps, baseline vs configuration (95% CI)
    figures/aggregate_<eval|train>.png        HNS mean / median / IQM / optimality gap with 95% CIs
    figures/ratio_heatmap_<eval|train>.png    per-game ratio (agent - random) / (baseline - random)
    figures/prob_improvement.png              P(agent > baseline) with 95% CIs
    figures/profiles_<eval|train>.png         performance profiles (fraction of runs with HNS > tau)
    cache.pkl                                 parsed logs (only new / modified logs are parsed again)

Usage:
    python stable_baselines3/plots/atari_results_tables.py
    python stable_baselines3/plots/atari_results_tables.py --results-dir <dir> --out-dir <dir> [--bootstrap 2000]
        [--workers 8] [--no-figures]
"""

import argparse
import math
import os
import pickle
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_RESULTS_DIR = r"C:\Users\sorte\Desktop\BIU\phd\suft2\ICML_2027\results"
KIND_DIRS = {"Baseline": "baseline", "SUFT": "suft"}
EXCLUDED_AGENTS = {"BBF", "DDPG", "DDPGSL", "SAC", "SACSL", "TD3", "TD3SL"}
# SUFT agent -> baseline agent
BASELINE_OF = {
    "DQNSL": "DQN",
    "DDQNSL": "DDQN",
    "DQN1SL": "DQN1",
    "DQNSLCER": "DQNCER",
    "DQNSLNStep": "DQNNStep",
    "DQNSLPERBoth": "DQNPER",
    "DQNSLPERLong": "DQNPER",
}
SCORES = {"eval": "final eval score", "train": "final training score"}

# Random and human scores of the 57 games (Badia et al., 2020, Agent57, Table "Atari 57 scores")
ATARI_RANDOM_HUMAN = {
    "Alien": (227.8, 7127.7), "Amidar": (5.8, 1719.5), "Assault": (222.4, 742.0),
    "Asterix": (210.0, 8503.3), "Asteroids": (719.1, 47388.7), "Atlantis": (12850.0, 29028.1),
    "BankHeist": (14.2, 753.1), "BattleZone": (2360.0, 37187.5), "BeamRider": (363.9, 16926.5),
    "Berzerk": (123.7, 2630.4), "Bowling": (23.1, 160.7), "Boxing": (0.1, 12.1),
    "Breakout": (1.7, 30.5), "Centipede": (2090.9, 12017.0), "ChopperCommand": (811.0, 7387.8),
    "CrazyClimber": (10780.5, 35829.4), "Defender": (2874.5, 18688.9), "DemonAttack": (152.1, 1971.0),
    "DoubleDunk": (-18.6, -16.4), "Enduro": (0.0, 860.5), "FishingDerby": (-91.7, -38.7),
    "Freeway": (0.0, 29.6), "Frostbite": (65.2, 4334.7), "Gopher": (257.6, 2412.5),
    "Gravitar": (173.0, 3351.4), "Hero": (1027.0, 30826.4), "IceHockey": (-11.2, 0.9),
    "Jamesbond": (29.0, 302.8), "Kangaroo": (52.0, 3035.0), "Krull": (1598.0, 2665.5),
    "KungFuMaster": (258.5, 22736.3), "MontezumaRevenge": (0.0, 4753.3), "MsPacman": (307.3, 6951.6),
    "NameThisGame": (2292.3, 8049.0), "Phoenix": (761.4, 7242.6), "Pitfall": (-229.4, 6463.7),
    "Pong": (-20.7, 14.6), "PrivateEye": (24.9, 69571.3), "Qbert": (163.9, 13455.0),
    "Riverraid": (1338.5, 17118.0), "RoadRunner": (11.5, 7845.0), "Robotank": (2.2, 11.9),
    "Seaquest": (68.4, 42054.7), "Skiing": (-17098.1, -4336.9), "Solaris": (1236.3, 12326.7),
    "SpaceInvaders": (148.0, 1668.7), "StarGunner": (664.0, 10250.0), "Surround": (-10.0, 6.5),
    "Tennis": (-23.8, -8.3), "TimePilot": (3568.0, 5229.2), "Tutankham": (11.4, 167.6),
    "UpNDown": (533.4, 11693.2), "Venture": (0.0, 1187.5), "VideoPinball": (0.0, 17667.9),
    "WizardOfWor": (563.5, 4756.5), "YarsRevenge": (3092.9, 54576.9), "Zaxxon": (32.5, 9173.3),
}

FILE_RE = re.compile(
    r"^(?P<env>(?P<game>[A-Za-z0-9]+)-v\d+)_(?P<agent>[A-Za-z0-9]+)_(?P<kind>baseline|suft)_(?P<cfg>.+)_run_(?P<run>\d+)\.log$")
EVAL_RE = re.compile(r"^Eval num_timesteps=(\d+), episode_reward=([-+\d.eE]+|nan)")
CFG_RES = {
    "lam": r"tf_lambda_([\d.]+)",
    "timesteps": r"timesteps_(\d+)",
    "buffer": r"(?<!short_)(?<!long_)buffer_(\d+)",
    "short": r"short_buffer_(\d+)",
    "long": r"long_buffer_(\d+)",
    "fraction": r"fraction_([\d.]+)",
    "n_steps": r"n_steps_(\d+)",
}
LABEL_FIELDS = {"lam": "λ", "short": "short", "fraction": "f", "n_steps": "n"}

BASE_COLOR, AGENT_COLOR = "#2a78d6", "#eb6834"
HUMAN_COLOR = "#d03b3b"  # human score reference line (a deep red, apart from the orange agent)
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
METRICS = [("median", "Median HNS"), ("iqm", "IQM HNS"), ("mean", "Mean HNS"), ("og", "Optimality gap")]


# ----------------------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------------------
def _cell(line: str) -> float:
    return float(line.split("|")[2])


def parse_log(path: str) -> dict:
    """Eval and training curves of a log, and whether it finished."""
    eval_t, eval_r, train_t, train_r = [], [], [], []
    rew, finished = None, False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("|    ep_rew_mean"):
                try:
                    rew = _cell(line)
                except ValueError:
                    rew = None
            elif line.startswith("|    total_timesteps"):
                if rew is not None:
                    train_t.append(_cell(line))
                    train_r.append(rew)
                rew = None
            elif line.startswith("-----"):
                rew = None
            elif line.startswith("Eval num_timesteps="):
                m = EVAL_RE.match(line)
                if m:
                    eval_t.append(float(m[1]))
                    eval_r.append(float(m[2]))
            elif line.startswith("Total train runtime"):
                finished = True
    return {"eval_t": np.asarray(eval_t), "eval_r": np.asarray(eval_r),
            "train_t": np.asarray(train_t), "train_r": np.asarray(train_r), "finished": finished}


def scan_logs(results_dir: str) -> list[str]:
    paths = []
    for kind_dir in KIND_DIRS:
        base = os.path.join(results_dir, kind_dir)
        if not os.path.isdir(base):
            continue
        for agent_dir in sorted(os.listdir(base)):
            if agent_dir in EXCLUDED_AGENTS or not os.path.isdir(os.path.join(base, agent_dir)):
                continue
            for root, _, files in os.walk(os.path.join(base, agent_dir)):
                paths += [os.path.join(root, f) for f in files if f.endswith(".log")]
    return sorted(paths)


def load_parsed(paths: list[str], cache_path: str, workers: int) -> dict[str, dict]:
    """Parse the logs, reusing the cache for the logs whose size and modification time did not change."""
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:  # corrupted / incompatible cache
            cache = {}
    out, todo = {}, []
    for p in paths:
        st = os.stat(p)
        key = (st.st_size, st.st_mtime_ns)
        if p in cache and cache[p][0] == key:
            out[p] = cache[p][1]
        else:
            todo.append((p, key))
    if todo:
        print(f"parsing {len(todo)} logs ({len(out)} from the cache) ...")
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                results = list(ex.map(parse_log, [p for p, _ in todo], chunksize=8))
        else:
            results = [parse_log(p) for p, _ in todo]
        for (p, key), r in zip(todo, results):
            out[p] = r
            cache[p] = (key, r)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    return out


def parse_cfg(cfg: str) -> dict:
    out = {}
    for key, pattern in CFG_RES.items():
        m = re.search(pattern, cfg)
        out[key] = float(m[1]) if m else np.nan
    m = re.search(r"cer_(.+)$", cfg)
    out["cer"] = m[1] if m else ""
    return out


def fmt(v) -> str:
    """1000000 -> 1M, 4000 -> 4K, 0.5 -> 0.5, nan -> ''."""
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    for scale, suffix in ((1e6, "M"), (1e3, "K")):
        if abs(v) >= scale:
            return f"{v / scale:g}{suffix}"
    return f"{v:g}"


# ----------------------------------------------------------------------------------------------------------
# Tables of runs / configurations
# ----------------------------------------------------------------------------------------------------------
def build_runs(parsed: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for path, d in parsed.items():
        m = FILE_RE.match(os.path.basename(path))
        if not m or m["game"] not in ATARI_RANDOM_HUMAN:
            print(f"skip (not an Atari-57 log): {os.path.basename(path)}")
            continue
        cfg = parse_cfg(m["cfg"])
        has_eval, has_train = len(d["eval_t"]) > 0, len(d["train_t"]) > 0
        rows.append({
            "path": path, "config": f"{m['agent']}_{m['kind']}_{m['cfg']}", "agent": m["agent"], "kind": m["kind"],
            "env": m["env"], "game": m["game"], "run": int(m["run"]), **cfg,
            "finished": d["finished"], "n_evals": len(d["eval_t"]),
            "last_eval_step": d["eval_t"][-1] if has_eval else np.nan,
            "final_eval": d["eval_r"][-1] if has_eval else np.nan,
            "final_train": d["train_r"][-1] if has_train else np.nan,
            "last_train_step": d["train_t"][-1] if has_train else np.nan,
        })
    runs = pd.DataFrame(rows)
    runs["complete"] = runs["finished"] & (runs["last_eval_step"] == runs["timesteps"]) & runs["final_train"].notna()
    return runs.sort_values(["kind", "agent", "config", "env", "run"]).reset_index(drop=True)


def build_configs(runs: pd.DataFrame) -> pd.DataFrame:
    cols = ["agent", "kind", "lam", "timesteps", "buffer", "short", "long", "fraction", "n_steps", "cer"]
    cfgs = runs.groupby("config")[cols].first()
    stats = runs[runs.complete].groupby("config").agg(games=("env", "nunique"), runs=("path", "size"))
    cfgs = cfgs.join(stats).fillna({"games": 0, "runs": 0})
    cfgs["n_incomplete"] = runs[~runs.complete].groupby("config").size().reindex(cfgs.index).fillna(0).astype(int)

    labels = {}
    for agent, g in cfgs.groupby("agent"):
        varying = [c for c in LABEL_FIELDS if g[c].nunique(dropna=False) > 1]
        for cid, r in g.iterrows():
            labels[cid] = " ".join([agent] + [f"{LABEL_FIELDS[c]}={fmt(r[c])}" for c in varying])
    cfgs["label"] = pd.Series(labels)

    def same(a, b):
        return (a == b) or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))

    baseline, ablation = {}, {}
    for cid, r in cfgs[cfgs.kind == "suft"].iterrows():
        base_agent = BASELINE_OF.get(r.agent)
        cands = cfgs[(cfgs.agent == base_agent) & (cfgs.kind == "baseline") & (cfgs.timesteps == r.timesteps)]
        if len(cands) > 1:
            print(f"warning: {len(cands)} baselines for {cid}, using {cands.index[0]}")
        baseline[cid] = cands.index[0] if len(cands) else None
        if r.lam > 0:
            cands = cfgs[(cfgs.agent == r.agent) & (cfgs.lam == 0)]
            cands = [c for c, q in cands.iterrows()
                     if all(same(q[k], r[k]) for k in ("timesteps", "short", "long", "fraction", "n_steps", "cer"))]
            ablation[cid] = cands[0] if cands else None
    cfgs["baseline"] = pd.Series(baseline)
    cfgs["lambda0"] = pd.Series(ablation)

    # Order: every baseline followed by its SUFT configurations
    order = []
    for base in sorted(cfgs[cfgs.kind == "baseline"].index, key=lambda c: cfgs.label[c]):
        order.append(base)
        suft = cfgs[(cfgs.kind == "suft") & (cfgs.baseline == base)]
        order += sorted(suft.index, key=lambda c: (cfgs.lam[c], cfgs.fraction[c], cfgs.label[c]))
    order += [c for c in cfgs.index if c not in order]
    return cfgs.loc[order]


# ----------------------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------------------
def score_matrix(runs: pd.DataFrame, config: str, col: str, envs: list[str]) -> np.ndarray:
    """(runs, games) matrix of a configuration; games with fewer runs are truncated to the smallest count."""
    sub = runs[(runs.config == config) & runs.complete]
    cols = [sub.loc[sub.env == e].sort_values("run")[col].to_numpy(dtype=float) for e in envs]
    n = min(len(c) for c in cols)
    if n == 0:
        return np.full((0, len(envs)), np.nan)
    if any(len(c) != n for c in cols):
        print(f"warning: {config} {col}: unequal number of runs per game, using {n} per game")
    return np.stack([c[:n] for c in cols], axis=1)


def agg_metrics(S: np.ndarray) -> dict[str, np.ndarray]:
    """rliable aggregates of HNS ``S`` of shape (..., runs, games)."""
    per_game = S.mean(-2)
    flat = np.sort(S.reshape(*S.shape[:-2], -1), axis=-1)
    n = flat.shape[-1]
    k = int(0.25 * n)  # scipy.stats.trim_mean(..., 0.25)
    return {
        "mean": per_game.mean(-1),
        "median": np.median(per_game, axis=-1),
        "iqm": flat[..., k:n - k].mean(-1),
        "og": (1.0 - np.minimum(flat, 1.0)).mean(-1),
    }


def stratified_bootstrap(S: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """(n_boot, runs, games[, ...]): runs resampled with replacement inside every game."""
    R, G = S.shape[:2]
    idx = rng.integers(0, R, size=(n_boot, R, G))
    return S[idx, np.arange(G)]


def metrics_with_ci(S: np.ndarray, n_boot: int, rng) -> dict[str, tuple[float, float, float]]:
    point = agg_metrics(S)
    boot = agg_metrics(stratified_bootstrap(S, n_boot, rng))
    return {k: (float(point[k]), *np.percentile(boot[k], [2.5, 97.5])) for k in point}


def prob_improvement(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """P(X > Y) (+ 0.5 P(X = Y)) over the run pairs of every game, mean over games. X, Y: (..., runs, games)."""
    gt = (X[..., :, None, :] > Y[..., None, :, :]).mean(axis=(-3, -2))
    eq = (X[..., :, None, :] == Y[..., None, :, :]).mean(axis=(-3, -2))
    return (gt + 0.5 * eq).mean(-1)


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilcoxon(d: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank test (normal approximation with tie correction), zeros dropped."""
    d = d[np.isfinite(d) & (np.abs(d) > 1e-12)]
    n = len(d)
    if n < 2:
        return 1.0
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_plus = ranks[d > 0].sum()
    _, counts = np.unique(ranks, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (counts ** 3 - counts).sum() / 48
    if var <= 0:
        return 1.0
    z = (w_plus - n * (n + 1) / 4) / math.sqrt(var)
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


# ----------------------------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------------------------
def analyze(runs, cfgs, envs, n_boot, seed):
    rnd = np.array([ATARI_RANDOM_HUMAN[e.split("-v")[0]][0] for e in envs])
    hum = np.array([ATARI_RANDOM_HUMAN[e.split("-v")[0]][1] for e in envs])
    rng = np.random.default_rng(seed)

    raw = {(c, s): score_matrix(runs, c, f"final_{s}", envs) for c in cfgs.index for s in SCORES}
    hns = {k: (v - rnd) / (hum - rnd) for k, v in raw.items()}

    # Per configuration x game
    per_game = []
    for (c, s), M in raw.items():
        if len(M) == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            std = M.std(0, ddof=1) if len(M) > 1 else np.full(len(envs), np.nan)
        for g, e in enumerate(envs):
            per_game.append({"config": c, "label": cfgs.label[c], "score": s, "env": e, "n": len(M),
                             "mean": M[:, g].mean(), "std": std[g], "sem": std[g] / math.sqrt(len(M)),
                             "random": rnd[g], "human": hum[g], "hns": hns[(c, s)][:, g].mean()})
    per_game = pd.DataFrame(per_game)

    # Aggregates per configuration
    aggregate = []
    for c in cfgs.index:
        for s in SCORES:
            if len(hns[(c, s)]) == 0:
                continue
            m = metrics_with_ci(hns[(c, s)], n_boot, rng)
            row = {"config": c, "label": cfgs.label[c], "kind": cfgs.kind[c], "score": s, "runs": len(hns[(c, s)])}
            for k, (v, lo, hi) in m.items():
                row.update({k: v, f"{k}_lo": lo, f"{k}_hi": hi})
            aggregate.append(row)
    aggregate = pd.DataFrame(aggregate)

    # Comparisons: SUFT vs baseline, and SUFT (lambda > 0) vs the same agent at lambda = 0
    pairs = [("baseline", c, cfgs.baseline[c]) for c in cfgs.index if cfgs.kind[c] == "suft" and isinstance(cfgs.baseline[c], str)]
    pairs += [("lambda0", c, cfgs.lambda0[c]) for c in cfgs.index if isinstance(cfgs.lambda0.get(c), str)]
    comparison, per_game_cmp = [], []
    for ref_kind, c, b in pairs:
        for s in SCORES:
            A, B = raw[(c, s)], raw[(b, s)]
            if len(A) == 0 or len(B) == 0:
                continue
            a_mean, b_mean = A.mean(0), B.mean(0)
            num, den = a_mean - rnd, b_mean - rnd
            valid = (num > 1) & (den > 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = num / den
            ha, hb = hns[(c, s)], hns[(b, s)]
            d_hns = ha.mean(0) - hb.mean(0)
            wins, losses = int((a_mean > b_mean).sum()), int((a_mean < b_mean).sum())
            pa, pb = agg_metrics(ha), agg_metrics(hb)
            boot_a, boot_b = stratified_bootstrap(ha, n_boot, rng), stratified_bootstrap(hb, n_boot, rng)
            ba, bb = agg_metrics(boot_a), agg_metrics(boot_b)
            p_imp = float(prob_improvement(A, B))
            p_imp_boot = prob_improvement(stratified_bootstrap(A, n_boot, rng), stratified_bootstrap(B, n_boot, rng))
            row = {
                "comparison": ref_kind, "config": c, "label": cfgs.label[c], "reference": b,
                "reference_label": cfgs.label[b], "score": s, "games": len(envs), "valid_games": int(valid.sum()),
                "ratio_pct": 100 * ratio[valid].mean() if valid.any() else np.nan,
                "median_ratio_pct": 100 * np.median(ratio[valid]) if valid.any() else np.nan,
                "wins": wins, "ties": len(envs) - wins - losses, "losses": losses,
                "sign_p": sign_test(wins, losses), "wilcoxon_p": wilcoxon(d_hns),
                "p_improvement": p_imp, "p_improvement_lo": np.percentile(p_imp_boot, 2.5),
                "p_improvement_hi": np.percentile(p_imp_boot, 97.5),
            }
            row["improvement_pct"] = row["ratio_pct"] - 100
            for k in pa:
                delta = ba[k] - bb[k]
                row.update({f"ref_{k}": float(pb[k]), f"agent_{k}": float(pa[k]), f"delta_{k}": float(pa[k] - pb[k]),
                            f"delta_{k}_lo": np.percentile(delta, 2.5), f"delta_{k}_hi": np.percentile(delta, 97.5)})
            comparison.append(row)
            for g, e in enumerate(envs):
                per_game_cmp.append({
                    "comparison": ref_kind, "config": c, "label": cfgs.label[c], "reference_label": cfgs.label[b],
                    "score": s, "env": e, "random": rnd[g], "human": hum[g], "reference_mean": b_mean[g],
                    "agent_mean": a_mean[g], "ratio": ratio[g], "valid": bool(valid[g]),
                    "reference_hns": hb.mean(0)[g], "agent_hns": ha.mean(0)[g]})
    return per_game, aggregate, pd.DataFrame(comparison), pd.DataFrame(per_game_cmp), (rnd, hum)


# ----------------------------------------------------------------------------------------------------------
# Curves
# ----------------------------------------------------------------------------------------------------------
def curve_grid(runs, parsed, which, step):
    ok = runs[runs.complete]
    if which == "eval":
        return np.unique(np.concatenate([parsed[p]["eval_t"] for p in ok.path]))
    return np.arange(step, ok.timesteps.max() + 1, step, dtype=float)


def curve_tensor(runs, parsed, config, envs, which, grid) -> np.ndarray:
    """(runs, games, steps) curves of a configuration on ``grid`` (nan before the first logged point)."""
    sub = runs[(runs.config == config) & runs.complete]
    run_ids = sorted(sub.run.unique())
    out = np.full((len(run_ids), len(envs), len(grid)), np.nan)
    for r in sub.itertuples():
        d = parsed[r.path]
        t, y = d[f"{which}_t"], d[f"{which}_r"]
        if len(t) == 0:
            continue
        right = np.nan if which == "eval" else y[-1]
        out[run_ids.index(r.run), envs.index(r.env)] = np.interp(grid, t, y, left=np.nan, right=right)
    return out


def mean_sem(y: np.ndarray, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n = np.sum(np.isfinite(y), axis=axis)
        m = np.nanmean(y, axis=axis)
        se = np.nanstd(y, axis=axis, ddof=1) / np.sqrt(n)
    return m, np.nan_to_num(se)


# ----------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------
def style(ax, fontsize=7):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(which="both", colors=TEXT_2, labelsize=fontsize)


def save(fig, path):
    if os.path.exists(path):  # overwriting in place can fail on Windows when the file is open / synced
        os.remove(path)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_games(envs, rnd, hum, tensors, labels, grid, title, path):
    n, ncols = len(envs), 8
    nrows = math.ceil((n + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 1.95), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    x = grid / 1e6
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        style(ax, 6)
        for T, color in zip(tensors, (BASE_COLOR, AGENT_COLOR)):
            m, se = mean_sem(T[:, i])
            ax.fill_between(x, m - se, m + se, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, m, color=color, linewidth=1.2, marker="o" if len(x) <= 20 else None, markersize=2)
        ax.axhline(rnd[i], color=TEXT_2, linewidth=0.8, linestyle="--")
        ax.axhline(hum[i], color=HUMAN_COLOR, linewidth=0.8, linestyle="--")
        ax.set_title(envs[i].split("-v")[0], fontsize=8, color=TEXT, pad=3)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        ax.yaxis.get_offset_text().set_fontsize(6)
        if i >= n - ncols:
            ax.set_xlabel("steps (M)", fontsize=6, color=TEXT_2)
    handles = [Line2D([], [], color=BASE_COLOR, linewidth=2, label=labels[0]),
               Line2D([], [], color=AGENT_COLOR, linewidth=2, label=labels[1]),
               Line2D([], [], color=TEXT_2, linewidth=1, linestyle="--", label="random"),
               Line2D([], [], color=HUMAN_COLOR, linewidth=1, linestyle="--", label="human")]
    axes.flat[n].legend(handles=handles, loc="center", frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, path)


def iqm_curve(S: np.ndarray, n_boot: int, rng):
    """IQM over (runs, games) at every step of S (runs, games, steps) with a 95% bootstrap CI."""
    keep = np.isfinite(S).all(axis=(0, 1))
    S = S[:, :, keep]

    def iqm(X):  # (..., runs, games, steps)
        flat = np.sort(X.reshape(*X.shape[:-3], -1, X.shape[-1]), axis=-2)
        n = flat.shape[-2]
        k = int(0.25 * n)
        return flat[..., k:n - k, :].mean(-2)

    boot = iqm(stratified_bootstrap(S, n_boot, rng))
    return keep, iqm(S), np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def plot_iqm_curves(pairs, curves, cfgs, grid, which, n_boot, rng, path):
    n = len(pairs)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.6, nrows * 3.2), dpi=130, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (c, b) in zip(axes.flat, pairs):
        style(ax)
        for cfg, color in ((b, BASE_COLOR), (c, AGENT_COLOR)):
            keep, m, lo, hi = iqm_curve(curves[cfg], n_boot, rng)
            x = grid[keep] / 1e6
            ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0)
            ax.plot(x, m, color=color, linewidth=1.6, label=cfgs.label[cfg],
                    marker="o" if len(x) <= 20 else None, markersize=3)
        ax.set_title(f"{cfgs.label[c]}  vs  {cfgs.label[b]}", fontsize=8, color=TEXT, loc="left")
        ax.set_xlabel("steps (M)", fontsize=7, color=TEXT_2)
        ax.set_ylabel("IQM human normalized score", fontsize=7, color=TEXT_2)
        ax.legend(frameon=False, fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"IQM HNS vs steps ({'evaluation' if which == 'eval' else 'training'} score, 95% stratified "
                 f"bootstrap CI)", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, path)


def plot_aggregate(aggregate, cfgs, score, path):
    agg = aggregate[aggregate.score == score].set_index("config").reindex([c for c in cfgs.index])
    agg = agg.dropna(subset=["iqm"])
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.36 * len(agg) + 1.4), dpi=130, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(agg))[::-1]
    for ax, (k, name) in zip(axes, METRICS):
        style(ax)
        for yi, (c, r) in zip(y, agg.iterrows()):
            color = BASE_COLOR if r.kind == "baseline" else AGENT_COLOR
            ax.plot([r[f"{k}_lo"], r[f"{k}_hi"]], [yi, yi], color=color, linewidth=6, alpha=0.45, solid_capstyle="butt")
            ax.plot([r[k]], [yi], marker="|", color=TEXT, markersize=10)
        ax.set_title(name, fontsize=9, color=TEXT)
        ax.set_xlabel("human normalized score", fontsize=7, color=TEXT_2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(agg.label, fontsize=7)
    fig.legend(handles=[Patch(color=BASE_COLOR, alpha=0.45, label="baseline"),
                        Patch(color=AGENT_COLOR, alpha=0.45, label="SUFT")],
               loc="upper right", frameon=False, fontsize=8, ncol=2)
    fig.suptitle(f"Aggregate HNS, {SCORES[score]} (point estimate and 95% CI)", fontsize=10, color=TEXT, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, path)


def plot_ratio_heatmap(per_game_cmp, envs, score, path):
    sub = per_game_cmp[(per_game_cmp.score == score) & (per_game_cmp.comparison == "baseline")]
    cols = list(dict.fromkeys(sub.label))
    ratio = sub.pivot_table(index="env", columns="label", values="ratio").reindex(index=envs, columns=cols)
    valid = sub.pivot_table(index="env", columns="label", values="valid").reindex(index=envs, columns=cols)
    data = np.where(valid.to_numpy(dtype=bool), np.log2(ratio.to_numpy().clip(1e-6)), np.nan)
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 3, 0.24 * len(envs) + 2.2), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor("#d9d8d4")
    im = ax.imshow(np.clip(data, -2, 2), cmap="RdBu", vmin=-2, vmax=2, aspect="auto")
    for i in range(len(envs)):
        for j in range(len(cols)):
            v = ratio.iat[i, j]
            if np.isfinite(v) and valid.iat[i, j]:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(data[i, j]) > 1.3 else TEXT)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([e.split("-v")[0] for e in envs], fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("log2 ratio (agent − random) / (baseline − random)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title(f"Per-game ratio vs baseline, {SCORES[score]}  (gray: not valid)", fontsize=9, color=TEXT, loc="left")
    fig.tight_layout()
    save(fig, path)


def plot_prob_improvement(comparison, path):
    sub = comparison[comparison.comparison == "baseline"]
    labels = list(dict.fromkeys(sub.label))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.4 * len(labels) + 1.4), dpi=130, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(labels))[::-1]
    for ax, s in zip(axes, SCORES):
        style(ax)
        r = sub[sub.score == s].set_index("label").reindex(labels)
        ax.axvline(0.5, color=TEXT_2, linewidth=1, linestyle="--")
        ax.hlines(y, r.p_improvement_lo, r.p_improvement_hi, color=AGENT_COLOR, linewidth=6, alpha=0.45)
        ax.plot(r.p_improvement, y, "|", color=TEXT, markersize=10)
        ax.set_title(f"P(SUFT > baseline), {SCORES[s]}", fontsize=9, color=TEXT)
        ax.set_xlabel("probability of improvement", fontsize=7, color=TEXT_2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{l}  vs  {sub[sub.label == l].reference_label.iloc[0]}" for l in labels], fontsize=7)
    fig.tight_layout()
    save(fig, path)


def plot_profiles(pairs, hns_final, cfgs, score, path):
    n = len(pairs)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.0), dpi=130, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    all_hns = np.concatenate([hns_final[(c, score)].ravel() for p in pairs for c in p])
    tau = np.linspace(0, max(1.0, np.percentile(all_hns, 97)), 200)
    for ax, (c, b) in zip(axes.flat, pairs):
        style(ax)
        for cfg, color in ((b, BASE_COLOR), (c, AGENT_COLOR)):
            S = hns_final[(cfg, score)].ravel()
            ax.plot(tau, (S[None, :] > tau[:, None]).mean(1), color=color, linewidth=1.6, label=cfgs.label[cfg])
        ax.set_title(f"{cfgs.label[c]}  vs  {cfgs.label[b]}", fontsize=8, color=TEXT, loc="left")
        ax.set_xlabel("human normalized score τ", fontsize=7, color=TEXT_2)
        ax.set_ylabel("fraction of runs with score > τ", fontsize=7, color=TEXT_2)
        ax.legend(frameon=False, fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"Performance profiles, {SCORES[score]}", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, path)


# ----------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def ci(v, lo, hi, digits=3):
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def pval(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def comparison_rows(comparison, kind, score):
    sub = comparison[(comparison.comparison == kind) & (comparison.score == score)]
    rows = []
    for r in sub.itertuples():
        rows.append({
            "Agent": r.label, "Reference": r.reference_label,
            "Ratio %": f"{r.ratio_pct:.1f}", "Improvement %": f"{r.improvement_pct:+.1f}",
            "Median ratio %": f"{r.median_ratio_pct:.1f}", "Valid games": r.valid_games,
            "W/T/L": f"{r.wins}/{r.ties}/{r.losses}", "Sign p": pval(r.sign_p), "Wilcoxon p": pval(r.wilcoxon_p),
            "P(improvement)": ci(r.p_improvement, r.p_improvement_lo, r.p_improvement_hi, 2),
            "Mean HNS (ref → agent)": f"{r.ref_mean:.3f} → {r.agent_mean:.3f}",
            "Median HNS": f"{r.ref_median:.3f} → {r.agent_median:.3f}",
            "IQM HNS": f"{r.ref_iqm:.3f} → {r.agent_iqm:.3f}",
            "Opt. gap": f"{r.ref_og:.3f} → {r.agent_og:.3f}",
            "ΔIQM [95% CI]": ci(r.delta_iqm, r.delta_iqm_lo, r.delta_iqm_hi),
        })
    return pd.DataFrame(rows)


def write_latex(comparison, path):
    sub = comparison[comparison.comparison == "baseline"]
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{SUFT vs.\ baseline on Atari-57 (1M steps). Ratio: $\frac{1}{|V|}\sum_{g\in V}"
             r"\frac{\text{agent}_g-\text{random}_g}{\text{baseline}_g-\text{random}_g}$ over the valid games $V$. "
             r"HNS: human normalized score (baseline $\rightarrow$ SUFT). W/L: games where SUFT is better / worse.}",
             r"\label{tab:atari57}", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Score & Agent & Ratio (\%) & $|V|$ & W/L & Mean HNS & Median HNS & IQM HNS & Opt.\ gap \\",
             r"\midrule"]
    for s in SCORES:
        for r in sub[sub.score == s].itertuples():
            lines.append(
                f"{'Eval' if s == 'eval' else 'Train'} & {r.label} vs {r.reference_label} & {r.ratio_pct:.1f} & "
                f"{r.valid_games} & {r.wins}/{r.losses} & {r.ref_mean:.3f}$\\rightarrow${r.agent_mean:.3f} & "
                f"{r.ref_median:.3f}$\\rightarrow${r.agent_median:.3f} & "
                f"{r.ref_iqm:.3f}$\\rightarrow${r.agent_iqm:.3f} & {r.ref_og:.3f}$\\rightarrow${r.agent_og:.3f} \\\\")
        lines.append(r"\midrule" if s == "eval" else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).replace("λ", r"$\lambda$") + "\n")


def analysis_text(comparison, aggregate, cfgs):
    out = []
    main = comparison[comparison.comparison == "baseline"]
    for s in SCORES:
        sub = main[main.score == s]
        if sub.empty:
            continue
        better = sub[(sub.ratio_pct > 100)]
        iqm_sig = sub[sub.delta_iqm_lo > 0]
        iqm_sig_worse = sub[sub.delta_iqm_hi < 0]
        best = sub.loc[sub.ratio_pct.idxmax()]
        best_iqm = sub.loc[sub.delta_iqm.idxmax()]
        out.append(f"**{SCORES[s].capitalize()}**")
        out.append(f"- {len(better)}/{len(sub)} SUFT configurations have a score ratio above 100% vs their baseline "
                   f"(mean ratio over configurations {sub.ratio_pct.mean():.1f}%).")
        out.append(f"- Highest ratio: {best.label} vs {best.reference_label}, {best.ratio_pct:.1f}% over "
                   f"{best.valid_games} valid games (W/T/L {best.wins}/{best.ties}/{best.losses}).")
        out.append(f"- Largest IQM gain: {best_iqm.label}, IQM {best_iqm.ref_iqm:.3f} → {best_iqm.agent_iqm:.3f} "
                   f"(Δ {best_iqm.delta_iqm:+.3f}, 95% CI [{best_iqm.delta_iqm_lo:+.3f}, {best_iqm.delta_iqm_hi:+.3f}]).")
        out.append(f"- IQM difference with a 95% CI above 0: {', '.join(iqm_sig.label) or 'none'}; "
                   f"below 0: {', '.join(iqm_sig_worse.label) or 'none'}.")
        sig = sub[sub.wilcoxon_p < 0.05]
        out.append(f"- Wilcoxon p < 0.05 on the per-game HNS differences: "
                   f"{', '.join(f'{r.label} (p={pval(r.wilcoxon_p)}, {r.wins}W/{r.losses}L)' for r in sig.itertuples()) or 'none'}.")
        out.append("")
    abl = comparison[comparison.comparison == "lambda0"]
    if not abl.empty:
        out.append("**SUFT term ablation (λ > 0 vs λ = 0, same short / long term buffer)**")
        for r in abl.itertuples():
            out.append(f"- {SCORES[r.score]}: {r.label} vs {r.reference_label}: ratio {r.ratio_pct:.1f}%, "
                       f"W/T/L {r.wins}/{r.ties}/{r.losses}, IQM {r.ref_iqm:.3f} → {r.agent_iqm:.3f} "
                       f"(Δ 95% CI [{r.delta_iqm_lo:+.3f}, {r.delta_iqm_hi:+.3f}]), "
                       f"P(improvement) {r.p_improvement:.2f}.")
        out.append("")
    return "\n".join(out)


def write_report(path, runs, cfgs, envs, per_game, aggregate, comparison, per_game_cmp, figures):
    L = ["# Atari-57: baselines vs SUFT", ""]
    complete = runs[runs.complete]
    L += [f"{len(runs)} logs, {len(complete)} complete, {len(envs)} games, "
          f"{complete.groupby(['config', 'env']).size().min()}–{complete.groupby(['config', 'env']).size().max()} "
          f"runs per configuration and game, {fmt(complete.timesteps.max())} environment steps.", "",
          "- **Final eval score**: last evaluation (at the last step) of every run, averaged over the runs of a game.",
          "- **Final training score**: last `rollout/ep_rew_mean` (mean return of the last 100 training episodes, "
          "ε-greedy) of every run, averaged over the runs of a game. The logger prints it with 3 significant digits.",
          "- **Ratio %** = 100 / |valid| · Σ_valid (agent − random) / (baseline − random), "
          "valid: agent − random > 1 and baseline − random > 1. **Improvement %** = ratio − 100.",
          "- **HNS** = (score − random) / (human − random). Mean / median over games of the per-game mean; IQM and "
          "optimality gap (γ = 1) over all (run, game) scores; 95% CIs from a stratified bootstrap over runs (rliable).",
          "- **W/T/L**: games where the agent's mean is higher / equal / lower. Sign test: exact binomial on W vs L. "
          "Wilcoxon: signed-rank test on the per-game HNS differences. P(improvement): rliable probability of "
          "improvement.", ""]

    L += ["## Configurations", ""]
    t = cfgs.reset_index()
    L.append(md_table(pd.DataFrame({
        "Label": t.label, "Type": t.kind, "Agent": t.agent, "λ": t.lam.map(fmt), "Buffer": t.buffer.map(fmt),
        "Short": t.short.map(fmt), "Long": t.long.map(fmt), "Fraction": t.fraction.map(fmt),
        "n-step": t.n_steps.map(fmt), "CER": t.cer, "Steps": t.timesteps.map(fmt),
        "Baseline": t.baseline.map(lambda b: cfgs.label[b] if isinstance(b, str) else ""),
        "Games": t.games.astype(int), "Runs": t.runs.astype(int), "Incomplete": t.n_incomplete})))
    L.append("")

    L += ["## Analysis", "", analysis_text(comparison, aggregate, cfgs)]

    for s in SCORES:
        L += [f"## SUFT vs baseline: {SCORES[s]}", "", md_table(comparison_rows(comparison, "baseline", s)), ""]
    if (comparison.comparison == "lambda0").any():
        for s in SCORES:
            L += [f"## SUFT term ablation (λ > 0 vs λ = 0): {SCORES[s]}", "",
                  md_table(comparison_rows(comparison, "lambda0", s)), ""]

    for s in SCORES:
        a = aggregate[aggregate.score == s].set_index("config").reindex(cfgs.index).dropna(subset=["iqm"])
        L += [f"## Aggregate human normalized scores (95% CI): {SCORES[s]}", "", md_table(pd.DataFrame({
            "Configuration": a.label, "Type": a.kind, "Runs": a.runs.astype(int),
            "Mean": [ci(r.mean, r.mean_lo, r.mean_hi) for r in a.itertuples()],
            "Median": [ci(r.median, r.median_lo, r.median_hi) for r in a.itertuples()],
            "IQM": [ci(r.iqm, r.iqm_lo, r.iqm_hi) for r in a.itertuples()],
            "Optimality gap": [ci(r.og, r.og_lo, r.og_hi) for r in a.itertuples()]})), ""]

    for s in SCORES:
        sub = per_game[per_game.score == s]
        wide = sub.pivot_table(index="env", columns="config", values="mean").reindex(index=envs, columns=cfgs.index)
        sem = sub.pivot_table(index="env", columns="config", values="sem").reindex(index=envs, columns=cfgs.index)
        table = pd.DataFrame({"Game": [e.split("-v")[0] for e in envs],
                              "Random": [f"{ATARI_RANDOM_HUMAN[e.split('-v')[0]][0]:g}" for e in envs],
                              "Human": [f"{ATARI_RANDOM_HUMAN[e.split('-v')[0]][1]:g}" for e in envs]})
        for c in cfgs.index:
            if c in wide:
                table[cfgs.label[c]] = [f"{m:.1f} ± {e:.1f}" for m, e in zip(wide[c], sem[c])]
        L += [f"## Per-game {SCORES[s]} (mean ± s.e. over runs)", "", md_table(table), ""]

    for s in SCORES:
        sub = per_game_cmp[(per_game_cmp.score == s) & (per_game_cmp.comparison == "baseline")]
        ratio = sub.pivot_table(index="env", columns="label", values="ratio").reindex(index=envs)
        valid = sub.pivot_table(index="env", columns="label", values="valid").reindex(index=envs)
        cols = list(dict.fromkeys(sub.label))
        table = pd.DataFrame({"Game": [e.split("-v")[0] for e in envs]})
        for c in cols:
            table[c] = [f"{r:.2f}" if v else f"({r:.2f})" for r, v in zip(ratio[c], valid[c].astype(bool))]
        L += [f"## Per-game ratio (agent − random) / (baseline − random): {SCORES[s]}", "",
              "Values in parentheses are not valid (numerator or denominator ≤ 1) and are excluded from the ratio %.",
              "", md_table(table), ""]

    if figures:
        L += ["## Figures", ""] + [f"- `{os.path.relpath(f, os.path.dirname(path))}`" for f in figures]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", default=None, help="default: <results-dir>/Atari_analysis")
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap samples of the CIs")
    parser.add_argument("--curve-bootstrap", type=int, default=500, help="bootstrap samples of the IQM curves")
    parser.add_argument("--grid-step", type=int, default=10_000, help="steps between the points of the training curves")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="processes parsing the logs")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # the labels contain λ (cp1252 Windows consoles)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out_dir = args.out_dir or os.path.join(args.results_dir, "Atari_analysis")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    paths = scan_logs(args.results_dir)
    parsed = load_parsed(paths, os.path.join(out_dir, "cache.pkl"), args.workers)
    runs = build_runs(parsed)
    cfgs = build_configs(runs)
    incomplete = runs[~runs.complete]
    if len(incomplete):
        print(f"{len(incomplete)} incomplete runs excluded:")
        for p in incomplete.path:
            print(f"  {os.path.basename(p)}")
    envs = sorted(runs.env.unique())
    print(f"{len(runs)} logs, {len(cfgs)} configurations, {len(envs)} games")

    per_game, aggregate, comparison, per_game_cmp, (rnd, hum) = analyze(runs, cfgs, envs, args.bootstrap, args.seed)

    runs.drop(columns=["path"]).assign(log=runs.path.map(os.path.basename)).to_csv(
        os.path.join(out_dir, "runs.csv"), index=False)
    cfgs.to_csv(os.path.join(out_dir, "configs.csv"))
    per_game.to_csv(os.path.join(out_dir, "per_game.csv"), index=False)
    per_game_cmp.to_csv(os.path.join(out_dir, "per_game_comparison.csv"), index=False)
    comparison.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)
    aggregate.to_csv(os.path.join(out_dir, "aggregate.csv"), index=False)
    write_latex(comparison, os.path.join(out_dir, "comparison_table.tex"))

    figures = []
    if not args.no_figures:
        rng = np.random.default_rng(args.seed)
        pairs = [(c, cfgs.baseline[c]) for c in cfgs.index if cfgs.kind[c] == "suft" and isinstance(cfgs.baseline[c], str)]
        hns_final = {}
        for c in cfgs.index:
            for s in SCORES:
                hns_final[(c, s)] = (score_matrix(runs, c, f"final_{s}", envs) - rnd) / (hum - rnd)
        for which in SCORES:
            grid = curve_grid(runs, parsed, which, args.grid_step)
            curves = {c: curve_tensor(runs, parsed, c, envs, which, grid) for c in cfgs.index}
            name = "evaluation" if which == "eval" else "training"
            for c, b in pairs:
                slug = re.sub(r"[^A-Za-z0-9.]+", "_", cfgs.label[c].replace("λ", "lambda").replace("=", ""))
                path = os.path.join(fig_dir, f"games_{slug}_{which}.png")
                n_runs = len(curves[c])
                plot_games(envs, rnd, hum, (curves[b], curves[c]), (cfgs.label[b], cfgs.label[c]), grid,
                           f"{cfgs.label[c]} vs {cfgs.label[b]}: {name} score per game "
                           f"(mean ± s.e. over {n_runs} runs)", path)
                figures.append(path)
                print(f"  {os.path.basename(path)}")
            norm = {c: (T - rnd[None, :, None]) / (hum - rnd)[None, :, None] for c, T in curves.items()}
            path = os.path.join(fig_dir, f"iqm_curves_{which}.png")
            plot_iqm_curves(pairs, norm, cfgs, grid, which, args.curve_bootstrap, rng, path)
            figures.append(path)
            path = os.path.join(fig_dir, f"aggregate_{which}.png")
            plot_aggregate(aggregate, cfgs, which, path)
            figures.append(path)
            path = os.path.join(fig_dir, f"ratio_heatmap_{which}.png")
            plot_ratio_heatmap(per_game_cmp, envs, which, path)
            figures.append(path)
            path = os.path.join(fig_dir, f"profiles_{which}.png")
            plot_profiles(pairs, hns_final, cfgs, which, path)
            figures.append(path)
        path = os.path.join(fig_dir, "prob_improvement.png")
        plot_prob_improvement(comparison, path)
        figures.append(path)

    report = os.path.join(out_dir, "atari_results.md")
    write_report(report, runs, cfgs, envs, per_game, aggregate, comparison, per_game_cmp, figures)
    print(f"\n{len(figures)} figures, tables and report -> {out_dir}")


if __name__ == "__main__":
    main()

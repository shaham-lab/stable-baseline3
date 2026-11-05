"""
Bound figures of DQN1SB / DQN1SLSB: figures 2 and 5 per group (figure 5 split into short / long term for DQN1SLSB).

Groups (every buffer configuration of the group is a series of the same figure):
    DQN1SB   no SUFT (tf_lambda=0)   buffer 500 ... 1M
    DQN1SB   SUFT    (tf_lambda=1)   buffer 500 ... 1M
    DQN1SLSB no SUFT (tf_lambda=0)   short buffer 500 ... 256K
    DQN1SLSB SUFT    (tf_lambda=1)   short buffer 500 ... 256K

The color encodes the buffer size (DQN1SLSB: the short buffer size) on an ordinal ramp, small = dark,
large = light; the same size has the same color in every figure. Lines are also named by the legend and
by direct labels at their right end.

Each log has one line per gradient update printed by ``_print_cold_buffer_suft()``:

    actions taken 104 | timestep 104 | gradient updates 1 | ... | cold buffer SUFT mean 4.79e-05 | ...

Quantity -> log field (DQN1SLSB: the "current ..." fields exist for the short and the long term buffer,
drawn solid and dashed in the same color):

    Cold Q                       cold buffer SUFT mean
    distance cold Q              distance                      (||theta - theta_cold||)
    number of updates from cold  gradient updates
    SUFT Q                       current [short|long term] SUFT mean
    distance SUFT Q              current [short|long term] behavior distance mean   (nan unless track=true)
    number of updates from SUFT  current [short|long term] gradient updates difference max

The score comes from the SB3 logger tables of the same log:

    score                        rollout/ep_rew_mean   (mean of the last 100 training episodes)
    eval score                   eval/mean_reward      (evaluation episodes, drawn as markers)
    number of updates            train/n_updates       (of the same table)

Figures (Y vs X):
    2 Cold Q   vs number of updates from cold
    5 SUFT Q   vs number of updates from cold   (DQN1SLSB: one figure for the short term, one for the long term)
    8 score    vs number of updates   (eval score as markers)
    9 eval score vs number of updates

Every series is the running mean (in time order) of the per-update values; --points adds the raw values.

Output: <out-dir>/<agent>_<no_SUFT|SUFT>/<N>_<y>[_short_term|_long_term]_vs_<x>.png

Usage:
    python stable_baselines3/plots/bound_figures.py
    python stable_baselines3/plots/bound_figures.py --results-dir <dir> --out-dir <dir> [--linear-y] [--points]
"""

import argparse
import os
import re
import textwrap
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation, NullFormatter, SymmetricalLogLocator

DEFAULT_RESULTS_DIR = r"C:\Users\sorte\Desktop\BIU\phd\suft2\ICML_2027\results\Bound"

# Buffer size is ordinal: colors are sampled evenly from this colormap, one per distinct buffer size of all the
# logs (smallest -> start). The range stops at 0.78 so that the lightest color keeps >= 2:1 contrast on SURFACE
SERIES_CMAP, SERIES_CMAP_RANGE = "plasma", (0.0, 0.78)
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"

LINE_PREFIX = "actions taken "
FILE_RE = re.compile(r"^(?P<env>[A-Za-z0-9]+-v\d+)_(?P<agent>[A-Za-z0-9]+)_(?P<cfg>.+?)_run_(?P<run>\d+)\.log$")

LABELS = {
    "cold_q": "Cold Q  (cold buffer SUFT mean)",
    "cold_dist": "distance cold Q  (‖θ − θ_cold‖)",
    "cold_updates": "number of updates from cold  (x < 100: no-training steps)",
    "suft_q": "SUFT Q  (current SUFT mean)",
    "suft_dist": "distance SUFT Q  (behavior distance mean)",
    "suft_updates": "number of updates from SUFT  (max over minibatch)",
    "score": "score  (rollout ep_rew_mean, last 100 episodes)",
    "score_updates": "number of updates from cold  (x < 100: no-training steps)",
    "eval_score": "eval score  (eval mean_reward, 100 episodes)",
    "eval_score_updates": "number of updates from cold  (x < 100: no-training steps)",
}
# Y quantities spanning many orders of magnitude -> log scale (unless --linear-y)
LOG_Y = {"cold_q", "suft_q"}
# X quantities that count gradient updates: shifted after the no-training steps
UPDATE_KEYS = {"cold_updates", "suft_updates", "score_updates", "eval_score_updates"}
# Quantities shared by the short and long term of DQN1SLSB (one line per configuration)
SHARED_KEYS = {"cold_q", "cold_dist", "cold_updates", "score", "score_updates", "eval_score", "eval_score_updates"}
# Y quantities without the (i, 0) no-training points: the score is not 0 before training
NO_PRE_POINTS = {"score", "eval_score"}
# Y quantities whose figure has the random score line
SCORE_KEYS = {"score", "eval_score"}
# Series with at most this many points also get a marker on every point
MAX_POINTS_WITH_MARKERS = 50
# Y quantity -> (y, x) keys of extra values drawn as markers on top of the line
MARKERS = {"score": ("eval_score", "eval_score_updates", "eval mean reward")}
# Score of the random policy per env, drawn as a horizontal dashed line in the score figure (--random-score overrides)
RANDOM_SCORES = {"Alien-v5": 227.8}

# (number, y, x) of the figures (the numbers of the original list of 7 figures are kept)
FIGURES = [
    (2, "cold_q", "cold_updates"),
    (5, "suft_q", "cold_updates"),
    (8, "score", "score_updates"),
    (9, "eval_score", "eval_score_updates"),
]
TERM_STYLES = {"": "-", "short term": "-", "long term": "--"}


# ----------------------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------------------
def parse_log(path: str) -> dict[str, np.ndarray]:
    """
    Return {log field name: values} of all the per-gradient-update lines of a log, plus the score of the
    SB3 logger tables: "score" / "score updates" (rollout/ep_rew_mean) and "eval score" / "eval score updates"
    (eval/mean_reward), each with the train/n_updates of its table.
    """
    columns: dict[str, list[float]] = {}
    tables: list[dict[str, float]] = []
    table, section = {}, ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("-----"):  # opening / closing line of a logger table
                if table:
                    tables.append(table)
                table, section = {}, ""
                continue
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) == 2:
                    if cells[0].endswith("/"):
                        section = cells[0]
                    else:
                        try:
                            table[section + cells[0]] = float(cells[1])
                        except ValueError:
                            pass
                continue
            if not line.startswith(LINE_PREFIX):
                continue
            fields = {}
            for part in line.rstrip("\n").split(" | "):
                name, _, value = part.rpartition(" ")
                try:
                    fields[name] = float(value)
                except ValueError:
                    break
            else:
                for name, value in fields.items():
                    columns.setdefault(name, []).append(value)
    if table:
        tables.append(table)
    n = min((len(v) for v in columns.values()), default=0)
    data = {name: np.asarray(values[:n]) for name, values in columns.items()}
    if not data:
        return data
    for name, key in (("score", "rollout/ep_rew_mean"), ("eval score", "eval/mean_reward")):
        rows = [t for t in tables if key in t]
        data[name] = np.asarray([t[key] for t in rows], dtype=float)
        # A table printed before the first gradient update has no train/ section
        data[f"{name} updates"] = np.asarray([t.get("train/n_updates", 0.0) for t in rows], dtype=float)
    return data


def terms_of(data: dict[str, np.ndarray]) -> list[tuple[str, dict[str, np.ndarray]]]:
    """
    Split a parsed log into terms: one ("") for DQN1SB, "short term" + "long term" for DQN1SLSB.
    Every term maps the quantity keys of ``LABELS`` to arrays.
    """
    common = {
        "cold_q": data["cold buffer SUFT mean"],
        "cold_dist": data["distance"],
        "cold_updates": data["gradient updates"],
        "score": data["score"],
        "score_updates": data["score updates"],
        "eval_score": data["eval score"],
        "eval_score_updates": data["eval score updates"],
    }
    prefixes = [("", "current ")] if "current SUFT mean" in data else \
        [("short term", "current short term "), ("long term", "current long term ")]
    nan = np.full_like(common["cold_q"], np.nan)
    return [(term, {
        **common,
        "suft_q": data.get(f"{p}SUFT mean", nan),
        "suft_dist": data.get(f"{p}behavior distance mean", nan),
        "suft_updates": data.get(f"{p}gradient updates difference max", nan),
    }) for term, p in prefixes]


def config_of(file_name: str) -> dict:
    """Agent, tf_lambda, buffer sizes and run of a log, from its file name."""
    m = FILE_RE.match(file_name)
    if not m:
        raise ValueError(f"unexpected log name: {file_name}")
    cfg = m["cfg"]

    def num(key):
        found = re.search(rf"(?:^|_){key}_([\d.]+)", cfg)
        return float(found[1]) if found else None

    return {
        "env": m["env"], "agent": m["agent"], "run": int(m["run"]),
        "lam": num("tf_lambda"), "buffer": num("buffer"),
        "short": num("short_buffer"), "long": num("long_buffer"), "fraction": num("fraction"),
    }


def human(n: float) -> str:
    """1000000 -> 1M, 4000 -> 4K, 999500 -> 999.5K, 500 -> 500."""
    for scale, suffix in ((1e6, "M"), (1e3, "K")):
        if n >= scale:
            return f"{n / scale:g}{suffix}"
    return f"{n:g}"


# ----------------------------------------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------------------------------------
def running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Running mean over ``window`` consecutive values (time order)."""
    if window <= 1 or len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def label_line_ends(ax, ends, log_y):
    """
    Direct labels at the right of the axes, at the height of the last point of each line,
    pushed apart so that they never overlap. ``ends``: list of (y, text, color).
    """
    if not ends:
        return
    fig = ax.figure
    fig.canvas.draw()
    to_axes = ax.transAxes.inverted()
    min_gap = 9 / (ax.get_window_extent().height * 72 / fig.dpi)  # 9pt in axes fraction
    pos = []
    for y, text, color in ends:
        y_axes = to_axes.transform(ax.transData.transform((ax.get_xlim()[1], y)))[1]
        pos.append([min(max(y_axes, 0.0), 1.0), text, color])
    pos.sort(key=lambda p: p[0])
    for i in range(1, len(pos)):
        pos[i][0] = max(pos[i][0], pos[i - 1][0] + min_gap)
    overflow = pos[-1][0] - 1.0
    if overflow > 0:  # shift everything down, keeping the gaps
        for p in pos:
            p[0] -= overflow
    for y_axes, text, color in pos:
        ax.annotate(text, xy=(1.0, y_axes), xycoords="axes fraction", xytext=(14, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7, color=TEXT)
        ax.annotate("", xy=(1.0, y_axes), xycoords="axes fraction", xytext=(12, 0), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=2))


def plot_figure(series, y_key, x_key, title, out_path, log_y, points, max_points, y_label=None, no_train_steps=100,
                hline=None):
    """
    ``series``: list of (label, color, terms) where terms is the output of ``terms_of()``.
    ``hline``: optional (y, label) of a horizontal dashed reference line.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    shared = y_key in SHARED_KEYS and x_key in SHARED_KEYS
    multi_term = not shared and any(len(terms) > 1 for _, _, terms in series)
    n_pre = 0 if y_key in NO_PRE_POINTS else no_train_steps
    shift = no_train_steps - 1

    y_low, ends, has_markers = [], [], False
    for label, color, terms in series:
        # The cold quantities and the score are shared by the short and long term
        for term, s in terms[:1] if shared else terms:
            x, y = s[x_key], s[y_key]
            mask = np.isfinite(x) & np.isfinite(y)
            if log_y:
                mask &= y > 0
            x, y = x[mask], y[mask]
            if len(x) == 0:
                continue
            # The no-training steps before the first gradient update: (i, 0) for i = 0 .. no_train_steps - 1,
            # and the updates are shifted after them (gradient update 1 -> x = no_train_steps)
            if x_key in UPDATE_KEYS:
                x = x + shift
                pre_x = np.arange(n_pre, dtype=float)
            else:
                pre_x = np.zeros(n_pre)
            pre_y = np.zeros(n_pre)
            if points:
                stride = max(1, len(x) // max_points)
                ax.scatter(np.concatenate([pre_x, x[::stride]]), np.concatenate([pre_y, y[::stride]]),
                           s=2, color=color, alpha=0.12, linewidths=0, rasterized=True)
            # Running mean of the logged values only, the no-training points are drawn as they are
            window = max(1, len(x) // 300)
            y_trend = np.exp(running_mean(np.log(y), window)) if log_y else running_mean(y, window)
            ax.plot(np.concatenate([pre_x, running_mean(x, window)]), np.concatenate([pre_y, y_trend]),
                    color=color, linewidth=1.8, linestyle=TERM_STYLES[term] if multi_term else "-",
                    marker="o" if len(x) <= MAX_POINTS_WITH_MARKERS else None, markersize=3.5)
            if y_key in MARKERS:
                my_key, mx_key, _ = MARKERS[y_key]
                mx, my = s[mx_key], s[my_key]
                m_mask = np.isfinite(mx) & np.isfinite(my)
                if m_mask.any():
                    has_markers = True
                    ax.scatter(mx[m_mask] + (shift if x_key in UPDATE_KEYS else 0), my[m_mask], s=22,
                               facecolors=SURFACE, edgecolors=color, linewidths=1.4, zorder=3)
            y_low.append(np.percentile(y if points else y_trend, 0.1))
            end_label = f"{label.split()[-1]} · {term}" if multi_term else label
            ends.append((x[-1], y_trend[-1], end_label, color))

    if not ends:
        ax.text(0.5, 0.5, "no data in the logs for this quantity\n"
                          "(behavior distance is only logged with track_behavior_distance=True)",
                transform=ax.transAxes, ha="center", va="center", color=TEXT_2, fontsize=9)
    elif log_y:
        # Symlog: linear from 0 to the smallest values (so that y = 0 is on the axis), log above.
        # The linear part ends at the 0.1% lowest value: a few ~0 values (e.g. the very first update)
        # would otherwise stretch the log part over 15 decades
        linthresh = min(y_low)
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
        high = ax.get_ylim()[1]
        ax.yaxis.set_minor_locator(SymmetricalLogLocator(base=10, linthresh=linthresh, subs=np.arange(2, 10)))
        if high / linthresh < 100:  # under two decades: label the minor ticks too, or there is ~1 tick label
            log_formatter = LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 2))
            # Only in the log part: below linthresh the ticks are squeezed together
            ax.yaxis.set_minor_formatter(FuncFormatter(
                lambda v, pos: log_formatter(v, pos) if v >= linthresh else ""))
        else:
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(bottom=-0.03 * linthresh, top=high)
    if ends and hline is not None:
        ax.axhline(hline[0], color=TEXT_2, linewidth=1.2, linestyle="--", zorder=1)
    if ends:
        # Every line starts at (0, 0); a small margin keeps the y = 0 segment off the axis line
        ax.set_xlim(left=-0.01 * ax.get_xlim()[1])
        if not log_y:
            ax.set_ylim(bottom=-0.01 * ax.get_ylim()[1])

    ax.set_xlabel(LABELS[x_key], color=TEXT_2, fontsize=9)
    ax.set_ylabel(y_label or LABELS[y_key], color=TEXT_2, fontsize=9)
    ax.set_title("\n".join(w for line in title.split("\n") for w in textwrap.wrap(line, 100)), color=TEXT, fontsize=9, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(which="both", colors=TEXT_2, labelsize=8)

    if ends:
        handles = [Line2D([], [], color=color, linewidth=2, label=label) for label, color, _ in series]
        if multi_term:
            handles += [Line2D([], [], color=TEXT_2, linewidth=1.5, linestyle=TERM_STYLES[t], label=t)
                        for t in ("short term", "long term")]
        if has_markers:
            handles.append(Line2D([], [], color=TEXT_2, marker="o", markerfacecolor=SURFACE, linestyle="none",
                                  markersize=5, label=MARKERS[y_key][2]))
        if hline is not None:
            handles.append(Line2D([], [], color=TEXT_2, linewidth=1.2, linestyle="--", label=hline[1]))
        ax.legend(handles=handles, frameon=False, fontsize=7, labelcolor=TEXT,
                  loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=min(len(handles), 4))

    fig.tight_layout(rect=(0, 0, 0.88, 1))
    # Direct labels only for the lines that reach the right edge (the others are named by the legend)
    x_min, x_max = ax.get_xlim()
    label_line_ends(ax, [(y, t, c) for x, y, t, c in ends if x >= x_min + 0.9 * (x_max - x_min)], log_y)
    # Overwriting in place can fail on Windows (Errno 22) when the folder is synced / the file is open
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", default=None, help="default: <results-dir>/figures")
    parser.add_argument("--linear-y", action="store_true", help="linear Y axis for the Q plots too")
    parser.add_argument("--points", action="store_true", help="also draw the raw per-update values")
    parser.add_argument("--max-points", type=int, default=20_000, help="raw points drawn per series (--points)")
    parser.add_argument("--no-train-steps", type=int, default=100,
                        help="points (i, 0), i = 0 .. N-1, added before the first gradient update (learning_starts)")
    parser.add_argument("--random-score", type=float, default=None,
                        help="score of the random policy drawn in the score figure (default: per env, RANDOM_SCORES)")
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")

    logs = []
    for root, _, files in os.walk(args.results_dir):
        if os.path.abspath(root).startswith(os.path.abspath(out_dir)):
            continue
        logs += [os.path.join(root, f) for f in files if f.endswith(".log")]

    # (agent, tf_lambda) -> [(config, terms)]
    groups = defaultdict(list)
    for path in sorted(logs):
        name = os.path.basename(path)
        data = parse_log(path)
        if not data:
            print(f"skip (no per-update lines): {name}")
            continue
        cfg = config_of(name)
        groups[(cfg["agent"], cfg["lam"])].append((cfg, terms_of(data)))
        print(f"{len(data['gradient updates']):>7} updates  {name}")

    # Color by buffer size, shared by all the groups (a size has the same color in every figure)
    size_of = lambda cfg: cfg["short"] if cfg["short"] is not None else (cfg["buffer"] or 0)
    sizes = sorted({size_of(cfg) for members in groups.values() for cfg, _ in members})
    cmap = matplotlib.colormaps[SERIES_CMAP]
    size_color = {size: matplotlib.colors.to_hex(cmap(v))
                  for size, v in zip(sizes, np.linspace(*SERIES_CMAP_RANGE, len(sizes)))}

    n_figures = 0
    for (agent, lam), members in sorted(groups.items()):
        # Series in increasing buffer size, so that the colors follow the size
        members.sort(key=lambda m: (size_of(m[0]), m[0]["run"]))
        several_runs = len({m[0]["run"] for m in members}) > 1
        series = []
        for cfg, terms in members:
            label = f"short buffer {human(cfg['short'])}" if cfg["short"] is not None else f"buffer {human(cfg['buffer'])}"
            if several_runs:
                label += f" run {cfg['run']}"
            series.append((label, size_color[size_of(cfg)], terms))

        suft = "SUFT" if lam else "no_SUFT"
        cfg0 = members[0][0]
        title = f"{cfg0['env']}  {agent}  {'SUFT' if lam else 'no SUFT'} (tf_lambda={lam:g})"
        subtitle = "" if cfg0["fraction"] is None else \
            f"\nsampling fraction {cfg0['fraction']:g}, long buffer = 1M − short buffer"
        fig_dir = os.path.join(out_dir, f"{agent}_{suft}")
        os.makedirs(fig_dir, exist_ok=True)
        terms = [term for term, _ in series[0][2]]
        for i, y_key, x_key in FIGURES:
            log_y = y_key in LOG_Y and not args.linear_y
            if y_key in SHARED_KEYS or len(terms) == 1:
                out_path = os.path.join(fig_dir, f"{i}_{y_key}_vs_{x_key}.png")
                hline = None
                if y_key in SCORE_KEYS:
                    random_score = args.random_score if args.random_score is not None \
                        else RANDOM_SCORES.get(cfg0["env"])
                    if random_score is not None:
                        hline = (random_score, f"random score {random_score:g}")
                plot_figure(series, y_key, x_key, title + subtitle, out_path, log_y, args.points, args.max_points,
                            no_train_steps=args.no_train_steps, hline=hline)
                n_figures += 1
                continue
            # Short and long term SUFT Q: one figure each
            for j, term in enumerate(terms):
                term_series = [(label, color, [t[j]]) for label, color, t in series]
                out_path = os.path.join(fig_dir, f"{i}_{y_key}_{term.replace(' ', '_')}_vs_{x_key}.png")
                plot_figure(term_series, y_key, x_key, f"{title}  ·  {term}{subtitle}", out_path, log_y,
                            args.points, args.max_points, y_label=f"SUFT Q  (current {term} SUFT mean)",
                            no_train_steps=args.no_train_steps)
                n_figures += 1
        print(f"{agent} {suft}: {len(series)} configurations -> {fig_dir}")

    print(f"\n{n_figures} figures from {len(logs)} logs -> {out_dir}")


if __name__ == "__main__":
    main()

"""
generate_tables.py
------------------
Generates LaTeX tables from cross-validation results stored in dataset folders.

Expected folder structure:
    <input_dir>/
        dataset-64/
            val-metrics-stage2.json
            variables.json
        dataset-72/
            ...
        ...

Each val-metrics-stage2.json contains lists of 5 values (one per fold) for:
    accuracy, precision, recall, f1-score, true-negative-rate

Output: one .tex file per table, written to <output_dir>/
"""

import json
import os
import re
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------

# Metrics to include and their display names (edit order/subset freely)
METRICS = {
    "accuracy":          "Accuracy",
    "precision":         "Precision",
    "recall":            "Recall",
    "f1-score":          "F1-Score",
    "true-negative-rate":"TNR",
}

# Significance level
ALPHA = 0.05

# T-test alternative: "greater" → upper triangular (row > col)
#                     "two-sided" → full matrix
T_TEST_ALTERNATIVE = "greater"


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_datasets(input_dir: str) -> dict[int, dict]:
    """
    Scans input_dir for subfolders matching 'dataset-<int>'.
    Returns dict keyed by patch size (int), value is dict with:
        'metrics': {metric_name: [fold_0, ..., fold_4]}
        'variables': {training hyperparameters}
    Sorted by patch size ascending.
    """
    pattern = re.compile(r"^dataset-(\d+)$")
    datasets = {}

    for entry in os.scandir(input_dir):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if not m:
            continue
        size = int(m.group(1))

        metrics_path   = os.path.join(entry.path, "val-metrics-stage2.json")
        variables_path = os.path.join(entry.path, "variables.json")

        with open(metrics_path,   "r") as f:
            metrics = json.load(f)
        with open(variables_path, "r") as f:
            variables = json.load(f)

        datasets[size] = {"metrics": metrics, "variables": variables}

    return dict(sorted(datasets.items()))


# ===========================================================================
# STATISTICS HELPERS
# ===========================================================================

def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """Two-sided 95% CI using t-distribution (appropriate for small n)."""
    a = np.array(values)
    n = len(a)
    se = stats.sem(a)
    h  = se * stats.t.ppf(0.975, df=n - 1)
    return float(np.mean(a) - h), float(np.mean(a) + h)


def shapiro_wilk(values: list[float]) -> tuple[float, float]:
    """Returns (W, p-value)."""
    w, p = stats.shapiro(values)
    return float(w), float(p)


def ttest_independent(a: list[float], b: list[float],
                      alternative: str) -> float:
    """Welch's independent t-test. Returns p-value."""
    _, p = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
    return float(p)


# ===========================================================================
# LATEX HELPERS
# ===========================================================================

def bold(s: str) -> str:
    return rf"\textbf{{{s}}}"


def _write_tex(output_dir: str, filename: str, content: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Wrote: {path}")


def _write_figure_tex(directory: str, stem: str,
                      caption: str, label: str) -> None:
    """
    Writes a .tex file with a descriptive sentence + figure environment.
    The image is referenced by stem only (no path), resolving from the
    document root in LaTeX.

    stem   : filename without extension (e.g. "all_metrics")
    caption: figure caption text (may contain LaTeX markup)
    label  : LaTeX label (e.g. "fig:all_metrics")
    """
    if stem == "all_metrics":
        intro = (
            "Fig.~\\ref{" + label + "} shows the evolution of all validation metrics "
            "as a function of patch size, with error bars representing "
            "$\\pm$ one standard deviation across the 5 folds."
        )
    else:
        metric_name = caption.split(" vs")[0]
        intro = (
            "Fig.~\\ref{" + label + "} shows the evolution of the "
            "\\textit{" + metric_name + "} metric as a function of patch size, "
            "with error bars representing $\\pm$ one standard deviation across the 5 folds."
        )

    lines = [
        intro,
        "",
        "\\begin{figure}[!htb]",
        "    \\centering",
        "    \\includegraphics[width=\\linewidth]{" + stem + "}",
        "    \\caption{" + caption + "}",
        "    \\label{" + label + "}",
        "\\end{figure}",
    ]

    _write_tex(directory, stem + ".tex", "\n".join(lines))


# ===========================================================================
# TABLE 1 — Cross-validation results (mean ± std + 95% CI)
# ===========================================================================

def generate_cv_results_table(datasets: dict[int, dict],
                               output_dir: str) -> None:
    """
    One row per dataset size. Columns: patch size + one column per metric.
    Section 1: mean ± std
    Section 2: 95% CI [lo, hi]
    """
    sizes       = list(datasets.keys())
    metric_keys = list(METRICS.keys())
    col_headers = " & ".join(["Size (px)"] + [METRICS[k] for k in metric_keys])
    num_cols    = 1 + len(metric_keys)
    col_spec    = "l" + "c" * len(metric_keys)

    rows_mean = []
    rows_ci   = []

    for size in sizes:
        mdata = datasets[size]["metrics"]

        mean_cells = [str(size)]
        ci_cells   = [str(size)]

        for key in metric_keys:
            vals = mdata[key]
            mu   = np.mean(vals)
            sd   = np.std(vals, ddof=1)
            lo, hi = confidence_interval_95(vals)

            mean_cells.append(rf"{mu:.4f} $\pm$ {sd:.4f}")
            ci_cells.append(rf"[{lo:.4f}, {hi:.4f}]")

        rows_mean.append(" & ".join(mean_cells) + r" \\")
        rows_ci.append(  " & ".join(ci_cells)   + r" \\")

    intro = (
        r"Table~\ref{tab:cv_results} presents the 5-fold cross-validation results "
        r"for each dataset, reported as mean $\pm$ standard deviation and 95\% "
        r"confidence intervals across folds."
    )

    lines = [
        intro+"\n\n",
        r"\begin{table}[!htb]",
        r"\centering",
        r"\caption{Cross-validation results (mean $\pm$ std and 95\% CI) "
        r"per dataset patch size.}",
        r"\label{tab:cv_results}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        col_headers + r" \\",
        r"\midrule",
        *rows_mean,
        r"\midrule",
        rf"\multicolumn{{{num_cols}}}{{l}}{{Confidence intervals (95\%)}} \\",
        r"\midrule",
        *rows_ci,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    _write_tex(output_dir, "table_cv_results.tex", "\n".join(lines))


# ===========================================================================
# TABLE 2 — Shapiro-Wilk normality test
# ===========================================================================

def generate_shapiro_table(datasets: dict[int, dict],
                            output_dir: str) -> None:
    """
    Rows: dataset sizes. Columns: metrics.
    Cell value: W / p-value. Bold when p < ALPHA.
    """
    sizes       = list(datasets.keys())
    metric_keys = list(METRICS.keys())
    col_headers = " & ".join(["Size (px)"] + [METRICS[k] for k in metric_keys])
    col_spec    = "l" + "c" * len(metric_keys)

    rows = []
    for size in sizes:
        mdata = datasets[size]["metrics"]
        cells = [str(size)]
        for key in metric_keys:
            w, p = shapiro_wilk(mdata[key])
            cell = rf"{w:.3f} / {p:.3f}"
            if p < ALPHA:
                cell = bold(cell)
            cells.append(cell)
        rows.append(" & ".join(cells) + r" \\")

    note = (rf"\multicolumn{{{1 + len(metric_keys)}}}{{l}}"
            rf"{{\small Note: n=5 per group limits test power. "
            rf"Bold indicates rejection of normality ($p < {ALPHA}$).}}  \\")

    intro = (
        r"Table~\ref{tab:shapiro} reports the Shapiro-Wilk statistic ($W$) and "
        r"corresponding $p$-value for each dataset and metric. "
        r"Values in bold indicate rejection of the normality hypothesis ($p < 0.05$). "
        r"Note that with $n=5$ observations per group, the test has limited statistical power."
    )

    lines = [
        intro+"\n\n",
        r"\begin{table}[!htb]",
        r"\centering",
        r"\caption{Shapiro-Wilk normality test results ($W$ / $p$-value) "
        r"per dataset and metric.}",
        r"\label{tab:shapiro}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        col_headers + r" \\",
        r"\midrule",
        *rows,
        r"\midrule",
        note,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    _write_tex(output_dir, "table_shapiro.tex", "\n".join(lines))


# ===========================================================================
# TABLE 3 — Pairwise independent t-test matrix (one table per metric)
# ===========================================================================

def generate_ttest_tables(datasets: dict[int, dict],
                           output_dir: str) -> None:
    """
    One .tex file per metric. Matrix of p-values for the question:
        "Does mean(X) > mean(Y)?"  (Welch's one-sided independent t-test)

    Cell (X, Y) answers: is mean(X) > mean(Y)?

    Which pairs are computed is controlled by the condition on (X, Y):
        X < Y  → upper triangular
        X != Y → full matrix (swap X < Y for X != Y to toggle)

    Bold when p < ALPHA.
    """
    sizes       = list(datasets.keys())
    metric_keys = list(METRICS.keys())
    n           = len(sizes)

    for key in metric_keys:
        metric_label = METRICS[key]

        col_spec = "l" + "c" * n
        header   = " & ".join(["Size (px)"] + [str(s) for s in sizes]) + r" \\"

        rows = []
        for X in sizes:                                  # row
            vals_X = datasets[X]["metrics"][key]
            cells  = [str(X)]

            for Y in sizes:                              # column
                if X != Y:                                # ← change to X != Y for full matrix # X < Y
                    vals_Y = datasets[Y]["metrics"][key]
                    p      = ttest_independent(vals_X, vals_Y, alternative="greater")
                    cell   = rf"{p:.3f}"
                    if p < ALPHA:
                        cell = bold(cell)
                elif X == Y:
                    cell = r"---"
                else:
                    cell = ""                            # empty below diagonal

                cells.append(cell)

            rows.append(" & ".join(cells) + r" \\")

        num_cols = n + 1
        note = (rf"\multicolumn{{{num_cols}}}{{l}}"
                rf"{{\small Cell $(X, Y)$: one-sided Welch's $t$-test, $H_1$: mean$(X) >$ mean$(Y)$. "
                rf"Bold: $p < {ALPHA}$. Bonferroni-adjusted $\alpha$ = "
                rf"{ALPHA / (n*(n-1)//2):.4f}.}}  \\")

        filename = f"table_ttest_{key.replace('-','_')}.tex"
        intro = (
            rf"Table~\ref{{tab:ttest_{key.replace('-','_')}}} presents the $p$-values "
            rf"of pairwise one-sided Welch's $t$-tests for \textit{{{metric_label}}}. "
            r"Each cell $(X, Y)$ contains the $p$-value for the hypothesis that "
            r"the mean of dataset $X$ is greater than the mean of dataset $Y$. "
            rf"Values in bold indicate statistical significance at $\alpha = {ALPHA}$."
        )

        lines = [
            intro+"\n\n",
            r"\begin{table}[!htb]",
            r"\centering",
            rf"\caption{{Pairwise $t$-test $p$-values for \textit{{{metric_label}}}. "
            rf"Each cell $(X, Y)$ tests $H_1$: mean$(X) >$ mean$(Y)$.}}",
            rf"\label{{tab:ttest_{key.replace('-','_')}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\midrule",
            note,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        ttest_path = os.path.join(output_dir, "t_test")
        os.makedirs(ttest_path, exist_ok=True)

        _write_tex(ttest_path, filename, "\n".join(lines))


# ===========================================================================
# TABLE 4 — Training arguments (one .tex per dataset)
# ===========================================================================

def generate_train_args_tables(datasets: dict[int, dict],
                                output_dir: str) -> None:
    """
    One .tex file per dataset, saved under <output_dir>/train-arguments/dataset-X.tex
    Each file contains a two-column table (Parameter / Value) describing the
    training hyperparameters from variables.json, grouped into three sections:
        - General
        - Stage 1 (Transfer Learning)
        - Stage 2 (Fine-tuning)
    """

    # Display names and grouping for known keys in variables.json
    # key -> (display_name, group)
    # group: "general" | "stage1" | "stage2"
    PARAM_MAP = {
        "model":                 ("Model",               "general"),
        "image_size":            ("Image size",          "general"),
        "batch_size":            ("Batch size",          "general"),
        "my_seed":               ("Random seed",         "general"),
        "learning_rate_stage_1": ("Learning rate",       "stage1"),
        "epochs_stage_1":        ("Epochs",              "stage1"),
        "learning_rate_stage_2": ("Learning rate",       "stage2"),
        "epochs_stage_2":        ("Epochs",              "stage2"),
        "early_stop_patience":   ("Early stop patience", "stage2"),
    }

    SECTION_LABELS = {
        "general": "General",
        "stage1":  "Stage 1 --- Transfer Learning",
        "stage2":  "Stage 2 --- Fine-tuning",
    }

    train_args_path = os.path.join(output_dir, "train-arguments")

    for size, data in datasets.items():
        v = data["variables"]

        def fmt_value(key: str) -> str:
            val = v.get(key, "---")
            if key == "image_size" and isinstance(val, list):
                return rf"${val[0]} \times {val[1]}$"
            return str(val)

        rows = []
        current_group = None

        for key, (display_name, group) in PARAM_MAP.items():
            if key not in v:
                continue

            # Insert section header row when group changes
            if group != current_group:
                current_group = group
                label = SECTION_LABELS[group]
                rows.append(
                    rf"\multicolumn{{2}}{{l}}{{\textit{{{label}}}}} \\"
                )
                rows.append(r"\midrule")

            rows.append(rf"\quad {display_name} & {fmt_value(key)} \\")

        intro = (
            rf"Table~\ref{{tab:train_args_{size}}} summarizes the training "
            rf"configuration used for the dataset with ${size} \times {size}$ px patches, "
            r"including general hyperparameters and the settings for each of the "
            r"two training stages."
        )

        lines = [
            intro+"\n\n",
            r"\begin{table}[!htb]",
            r"\centering",
            rf"\caption{{Training configuration for the dataset with "
            rf"${size} \times {size}$ px patches.}}",
            rf"\label{{tab:train_args_{size}}}",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Parameter & Value \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        _write_tex(train_args_path, f"dataset-{size}.tex", "\n".join(lines))



# ===========================================================================
# PLOTS — metric scores vs patch size
# ===========================================================================

def generate_plots(datasets: dict[int, dict], output_dir: str) -> None:
    """
    Generates two sets of plots saved as PDF under <output_dir>/plots/:

    1. plots/all_metrics.pdf
       All metrics on a single figure, one line per metric.
       X-axis: patch size (px). Y-axis: score.
       Error bars: ± std across the 5 folds.

    2. plots/metrics/<metric_key>.pdf
       One figure per metric, showing mean ± std across patch sizes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes       = list(datasets.keys())
    metric_keys = list(METRICS.keys())

    # Pre-compute means and stds for every metric and size
    # means[key] = list of mean values, one per size (same order as sizes)
    means = {key: [] for key in metric_keys}
    stds  = {key: [] for key in metric_keys}

    for size in sizes:
        mdata = datasets[size]["metrics"]
        for key in metric_keys:
            vals = np.array(mdata[key])
            means[key].append(float(np.mean(vals)))
            stds[key].append(float(np.std(vals, ddof=1)))

    plots_dir   = os.path.join(output_dir, "plots")
    metrics_dir = os.path.join(plots_dir,  "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # ── Shared style ──────────────────────────────────────────────────────────
    STYLE = {
        "marker":    "o",
        "capsize":   4,
        "linewidth": 1.5,
        "markersize": 5,
    }

    # ── 1. All metrics on a single figure ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for key in metric_keys:
        ax.errorbar(
            sizes, means[key], yerr=stds[key],
            label=METRICS[key],
            **STYLE,
        )

    ax.set_xlabel("Patch size (px)")
    ax.set_ylabel("Score")
    ax.set_title("Validation metrics vs patch size")
    ax.set_xticks(sizes)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = os.path.join(plots_dir, "all_metrics.pdf")
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Wrote: {out_path}")

    _write_figure_tex(
        directory = plots_dir,
        stem      = "all_metrics",
        caption   = r"Validation metrics vs patch size. Each curve represents one metric; "
                    r"error bars indicate $\pm$ one standard deviation across the 5 folds.",
        label     = "fig:all_metrics",
    )

    # ── 2. Individual metric figures ──────────────────────────────────────────
    for key in metric_keys:
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.errorbar(
            sizes, means[key], yerr=stds[key],
            color="steelblue",
            label=METRICS[key],
            **STYLE,
        )

        ax.set_xlabel("Patch size (px)")
        ax.set_ylabel("Score")
        ax.set_title(f"{METRICS[key]} vs patch size")
        ax.set_xticks(sizes)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        fname    = f"{key.replace('-', '_')}.pdf"
        out_path = os.path.join(metrics_dir, fname)
        fig.savefig(out_path, format="pdf")
        plt.close(fig)
        print(f"  Wrote: {out_path}")

        _write_figure_tex(
            directory = metrics_dir,
            stem      = key.replace("-", "_"),
            caption   = f"{METRICS[key]} vs patch size. "
                        r"Error bars indicate $\pm$ one standard deviation across the 5 folds.",
            label     = f"fig:{key.replace('-', '_')}",
        )

# ===========================================================================
# MAIN
# ===========================================================================

def main():
    INPUT_DIR  = "/media/fernando/INFORMATION/KIDNEY/ResNet50/"          # directory containing dataset-X subfolders
    OUTPUT_DIR = "./output_tables"   # directory where .tex files will be saved
    
    print(f"Loading datasets from: {INPUT_DIR}")
    datasets = load_datasets(INPUT_DIR)
    
    print(f"Found {len(datasets)} datasets: sizes = {list(datasets.keys())}")

    print("\nGenerating tables...")
    generate_cv_results_table(datasets, OUTPUT_DIR)
    generate_shapiro_table(datasets, OUTPUT_DIR)
    generate_ttest_tables(datasets, OUTPUT_DIR)
    generate_train_args_tables(datasets, OUTPUT_DIR)
    generate_plots(datasets, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()

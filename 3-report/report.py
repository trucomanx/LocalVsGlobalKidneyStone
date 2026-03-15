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

    lines = [
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

    lines = [
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
        lines = [
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

    print("\nDone.")


if __name__ == "__main__":
    main()

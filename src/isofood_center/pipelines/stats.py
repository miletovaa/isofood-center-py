"""Standalone statistical tests & calculations — ported from the reference
peptide_analysis/lcms pipelines, generalized to any wide DataFrame (rows =
samples, columns = features) plus a group column.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, mannwhitneyu, probplot, shapiro, ttest_ind
from statsmodels.stats.multitest import multipletests


def descriptive_stats(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    """Mean/median/std/n of `value_col`, one row per `group_col` value."""
    return (
        df.groupby(group_col)[value_col]
        .agg(mean="mean", median="median", std="std", n="count")
        .reset_index()
    )


def rsd(df: pd.DataFrame, columns: list[str] | None = None) -> pd.Series:
    """Relative standard deviation (std/mean * 100) per column."""
    columns = columns or list(df.columns)
    X = df[columns].apply(pd.to_numeric, errors="coerce")
    return X.std() / X.mean() * 100


def shapiro_by_group(df: pd.DataFrame, value_col: str, group_col: str, alpha: float = 0.01) -> pd.DataFrame:
    """Shapiro-Wilk normality test, one row per group (skips groups with <3
    non-null values — the test isn't defined below that).
    """
    rows = []
    for group, sub in df.groupby(group_col):
        values = sub[value_col].dropna().astype(float).values
        if len(values) < 3:
            continue
        stat, pval = shapiro(values)
        rows.append({
            group_col: group,
            "n": len(values),
            "shapiro_stat": stat,
            "shapiro_pval": pval,
            "is_normal": pval > alpha,
        })
    return pd.DataFrame(rows)


def qq_regression(values) -> tuple[float, float, float]:
    """Q-Q plot regression against a normal distribution — a secondary
    normality check alongside Shapiro-Wilk.

    Returns:
        `(slope, intercept, r)` of the fitted line.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    _, (slope, intercept, r) = probplot(values)
    return slope, intercept, r


def _groups(df: pd.DataFrame, value_col: str, group_col: str) -> list[np.ndarray]:
    return [
        g[value_col].dropna().astype(float).values
        for _, g in df.groupby(group_col)
        if g[value_col].notna().any()
    ]


def kruskal_wallis(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """Kruskal-Wallis test across the groups in `group_col`."""
    stat, pval = kruskal(*_groups(df, value_col, group_col))
    return {"statistic": stat, "pvalue": pval}


def anova(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """One-way ANOVA across the groups in `group_col`."""
    stat, pval = f_oneway(*_groups(df, value_col, group_col))
    return {"statistic": stat, "pvalue": pval}


def bonferroni_correction(pvals) -> np.ndarray:
    """Bonferroni-corrected p-values."""
    _, corrected, _, _ = multipletests(pvals, method="bonferroni")
    return corrected


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg / FDR-corrected p-values."""
    _, corrected, _, _ = multipletests(pvals, method="fdr_bh")
    return corrected


def _run_group_test(method: str, groups: list[np.ndarray]) -> tuple[float, float]:
    if method == "anova":
        return f_oneway(*groups)
    if method == "kruskal":
        return kruskal(*groups)
    if method in ("welch", "mannwhitney"):
        if len(groups) != 2:
            raise ValueError(f"test={method!r} needs exactly 2 groups, got {len(groups)}")
        if method == "welch":
            res = ttest_ind(groups[0], groups[1], equal_var=False)
        else:
            res = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        return res.statistic, res.pvalue
    raise ValueError(f"unknown test {method!r}")


def test_all_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    test: Literal["auto", "kruskal", "anova", "welch", "mannwhitney"] = "auto",
    alpha: float = 0.01,
    correction: Literal["bonferroni", "fdr_bh", None] = "fdr_bh",
) -> pd.DataFrame:
    """Run a group-comparison test on every feature and correct across the
    resulting p-values — the per-feature driver every pipeline's stats
    sub-pipeline calls.

    Args:
        df: Wide DataFrame (rows = samples).
        feature_cols: Columns to test, one at a time.
        group_col: Column identifying each sample's group.
        test: "kruskal", "anova", "welch" or "mannwhitney" to force one
            test for every feature ("welch"/"mannwhitney" need exactly two
            groups); "auto" checks Shapiro normality (at `alpha`) in every
            group and then picks Welch's t-test / Mann-Whitney U for two
            groups, one-way ANOVA / Kruskal-Wallis for three or more
            (see docs/choosing_a_test.md, Results A and B).
        alpha: Significance threshold, applied both to the normality check
            (when `test="auto"`) and to `significant`.
        correction: Multiple-testing correction applied across all tested
            features; `None` skips correction (`pvalue_corrected == pvalue`).

    Returns:
        One row per tested feature, sorted by corrected p-value ascending.
    """
    rows = []
    for feature in feature_cols:
        groups = _groups(df, feature, group_col)
        if len(groups) < 2:
            continue

        if test == "auto":
            normal_in_all = all(len(g) >= 3 and shapiro(g)[1] > alpha for g in groups)
            if len(groups) == 2:
                method = "welch" if normal_in_all else "mannwhitney"
            else:
                method = "anova" if normal_in_all else "kruskal"
        else:
            method = test

        stat, pval = _run_group_test(method, groups)
        rows.append({"feature": feature, "test": method, "statistic": stat, "pvalue": pval})

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    if correction == "bonferroni":
        result["pvalue_corrected"] = bonferroni_correction(result["pvalue"].values)
    elif correction == "fdr_bh":
        result["pvalue_corrected"] = benjamini_hochberg(result["pvalue"].values)
    else:
        result["pvalue_corrected"] = result["pvalue"]

    result["significant"] = result["pvalue_corrected"] < alpha
    return result.sort_values("pvalue_corrected").reset_index(drop=True)


# --------------------------------------------------------------------------
# The methods below were added after the O2/ISOFood publication survey
# (method_statistics.csv): in order of how often O2 papers use them —
# correlation, linear regression, ANOVA + post-hoc, t-tests / Mann-Whitney,
# normality/variance tests, mixed models, paired designs, chi-square/Fisher.
# Each one maps to a result card in docs/choosing_a_test.md.
# --------------------------------------------------------------------------


def _is_normal(values: np.ndarray, alpha: float) -> bool:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return len(values) >= 3 and shapiro(values)[1] > alpha


def levene_test(df: pd.DataFrame, value_col: str, group_col: str, center: str = "median") -> dict:
    """Levene's (Brown-Forsythe with `center="median"`) test for equal
    variances across groups."""
    from scipy.stats import levene

    stat, pval = levene(*_groups(df, value_col, group_col), center=center)
    return {"statistic": stat, "pvalue": pval}


def welch_ttest(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """Welch's t-test between exactly two groups (Result A, parametric)."""
    stat, pval = _run_group_test("welch", _groups(df, value_col, group_col))
    return {"statistic": stat, "pvalue": pval}


def mann_whitney(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """Mann-Whitney U test between exactly two groups (Result A, non-parametric)."""
    stat, pval = _run_group_test("mannwhitney", _groups(df, value_col, group_col))
    return {"statistic": stat, "pvalue": pval}


def effect_size(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """Effect sizes to report next to a p-value (docs §6): Cohen's d and
    fold change (mean ratio) for two groups, eta² for three or more."""
    groups = _groups(df, value_col, group_col)
    if len(groups) == 2:
        a, b = groups
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / pooled if pooled else np.nan
        fold = a.mean() / b.mean() if b.mean() else np.nan
        return {"cohens_d": d, "fold_change": fold}
    all_values = np.concatenate(groups)
    grand = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((all_values - grand) ** 2).sum()
    return {"eta_squared": ss_between / ss_total if ss_total else np.nan}


def tukey_hsd(df: pd.DataFrame, value_col: str, group_col: str, alpha: float = 0.05) -> pd.DataFrame:
    """Tukey HSD post-hoc test: every pair of groups, family-wise corrected."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    sub = df[[value_col, group_col]].dropna()
    res = pairwise_tukeyhsd(sub[value_col].astype(float), sub[group_col].astype(str), alpha=alpha)
    table = pd.DataFrame(res._results_table.data[1:], columns=res._results_table.data[0])
    return table.rename(columns={"p-adj": "pvalue_adj", "meandiff": "mean_diff"}).assign(
        significant=lambda t: t["pvalue_adj"] < alpha
    )[["group1", "group2", "mean_diff", "pvalue_adj", "lower", "upper", "significant"]]


def dunn_test(
    df: pd.DataFrame, value_col: str, group_col: str, correction: str = "bonferroni", alpha: float = 0.05,
) -> pd.DataFrame:
    """Dunn's post-hoc test after Kruskal-Wallis (rank-based, tie-corrected),
    with p-values adjusted across all pairs (`correction` is any
    `statsmodels.stats.multitest.multipletests` method)."""
    from itertools import combinations

    from scipy.stats import norm, rankdata

    sub = df[[value_col, group_col]].dropna()
    values = sub[value_col].astype(float).values
    labels = sub[group_col].astype(str).to_numpy(dtype=str)
    ranks = rankdata(values)
    n = len(values)
    _, tie_counts = np.unique(values, return_counts=True)
    tie_term = (tie_counts ** 3 - tie_counts).sum() / (12 * (n - 1))

    rows = []
    for g1, g2 in combinations(sorted(set(labels)), 2):
        r1, r2 = ranks[labels == g1], ranks[labels == g2]
        se = np.sqrt((n * (n + 1) / 12 - tie_term) * (1 / len(r1) + 1 / len(r2)))
        z = (r1.mean() - r2.mean()) / se
        rows.append({"group1": g1, "group2": g2, "z": z, "pvalue": 2 * norm.sf(abs(z))})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["pvalue_adj"] = multipletests(result["pvalue"], method=correction)[1]
    result["significant"] = result["pvalue_adj"] < alpha
    return result


def posthoc(df: pd.DataFrame, value_col: str, group_col: str, test: str = "auto", alpha: float = 0.01) -> pd.DataFrame:
    """Which groups differ? Tukey HSD after ANOVA, Dunn after Kruskal-Wallis.
    `test="auto"` uses the same normality switch as `test_all_features`."""
    if test == "auto":
        normal = all(_is_normal(g, alpha) for g in _groups(df, value_col, group_col))
        test = "tukey" if normal else "dunn"
    table = tukey_hsd(df, value_col, group_col, alpha=alpha) if test == "tukey" else dunn_test(df, value_col, group_col, alpha=alpha)
    return table.assign(test=test)


def paired_test(
    df: pd.DataFrame, value_col: str, subject_col: str, condition_col: str, test: str = "auto", alpha: float = 0.01,
) -> dict:
    """Two measurements on the same samples (Result C): paired t-test or
    Wilcoxon signed-rank. Long-format input: one row per subject×condition.
    Subjects not measured under both conditions are dropped."""
    from scipy.stats import ttest_rel, wilcoxon

    wide = df.pivot_table(index=subject_col, columns=condition_col, values=value_col, aggfunc="mean").dropna()
    if wide.shape[1] != 2:
        raise ValueError(f"paired_test needs exactly 2 conditions, got {list(wide.columns)}")
    a, b = wide.iloc[:, 0].values, wide.iloc[:, 1].values
    if test == "auto":
        test = "paired_t" if _is_normal(b - a, alpha) else "wilcoxon"
    res = ttest_rel(a, b) if test == "paired_t" else wilcoxon(a, b)
    return {"test": test, "statistic": res.statistic, "pvalue": res.pvalue, "n_pairs": len(wide),
            "mean_diff": float((b - a).mean()), "conditions": tuple(wide.columns)}


def repeated_measures_test(
    df: pd.DataFrame, value_col: str, subject_col: str, condition_col: str, test: str = "auto", alpha: float = 0.01,
) -> dict:
    """Three or more measurements on the same samples (Result D):
    repeated-measures ANOVA or Friedman. Only complete subjects are used."""
    from scipy.stats import friedmanchisquare
    from statsmodels.stats.anova import AnovaRM

    wide = df.pivot_table(index=subject_col, columns=condition_col, values=value_col, aggfunc="mean").dropna()
    if test == "auto":
        test = "rm_anova" if all(_is_normal(wide[c].values, alpha) for c in wide.columns) else "friedman"
    if test == "friedman":
        stat, pval = friedmanchisquare(*[wide[c].values for c in wide.columns])
    else:
        long = wide.reset_index().melt(id_vars=subject_col, var_name=condition_col, value_name=value_col)
        table = AnovaRM(long, depvar=value_col, subject=subject_col, within=[condition_col]).fit().anova_table
        stat, pval = table["F Value"].iloc[0], table["Pr > F"].iloc[0]
    return {"test": test, "statistic": stat, "pvalue": pval, "n_subjects": len(wide), "conditions": list(wide.columns)}


def two_way_anova(
    df: pd.DataFrame, value_col: str, factor_a: str, factor_b: str, test: str = "auto", alpha: float = 0.01, typ: int = 2,
) -> pd.DataFrame:
    """Two grouping factors and their interaction (Result E): two-way ANOVA
    or the Aligned Rank Transform (ART) ANOVA when residuals aren't normal.

    Returns:
        One row per effect (`factor_a`, `factor_b`, interaction) with F and p.
    """
    import statsmodels.formula.api as smf
    from scipy.stats import rankdata
    from statsmodels.stats.anova import anova_lm

    sub = df[[value_col, factor_a, factor_b]].dropna().rename(columns={value_col: "y", factor_a: "A", factor_b: "B"})
    sub["y"] = sub["y"].astype(float)
    formula = "y ~ C(A) * C(B)"
    names = {"C(A)": factor_a, "C(B)": factor_b, "C(A):C(B)": f"{factor_a}:{factor_b}"}

    if test == "auto":
        resid = smf.ols(formula, data=sub).fit().resid
        test = "anova" if _is_normal(resid.values, alpha) else "art"

    if test == "anova":
        table = anova_lm(smf.ols(formula, data=sub).fit(), typ=typ)
        out = table.drop(index="Residual").rename(index=names)[["F", "PR(>F)"]]
    else:
        # ART (Wobbrock et al. 2011): for each effect, strip all other effects
        # from the response, rank, then read only that effect's F from a full
        # factorial ANOVA on the ranks.
        grand = sub["y"].mean()
        ma = sub.groupby("A")["y"].transform("mean")
        mb = sub.groupby("B")["y"].transform("mean")
        mab = sub.groupby(["A", "B"])["y"].transform("mean")
        resid = sub["y"] - mab
        aligned = {
            "C(A)": resid + (ma - grand),
            "C(B)": resid + (mb - grand),
            "C(A):C(B)": resid + (mab - ma - mb + grand),
        }
        rows = {}
        for effect, y_aligned in aligned.items():
            ranked = sub.assign(y=rankdata(y_aligned))
            table = anova_lm(smf.ols(formula, data=ranked).fit(), typ=typ)
            rows[names[effect]] = table.loc[effect, ["F", "PR(>F)"]]
        out = pd.DataFrame(rows).T
    out = out.rename(columns={"PR(>F)": "pvalue"}).rename_axis("effect").reset_index()
    out["test"] = test
    out["significant"] = out["pvalue"] < alpha
    return out


def correlation(x, y, method: str = "auto", alpha: float = 0.01) -> dict:
    """Correlation between two features (Result F): Pearson when both are
    normal, Spearman otherwise; `method` can also force "pearson",
    "spearman" or "kendall"."""
    from scipy.stats import kendalltau, pearsonr, spearmanr

    pair = pd.DataFrame({"x": pd.to_numeric(pd.Series(x), errors="coerce").values,
                         "y": pd.to_numeric(pd.Series(y), errors="coerce").values}).dropna()
    if method == "auto":
        method = "pearson" if _is_normal(pair["x"].values, alpha) and _is_normal(pair["y"].values, alpha) else "spearman"
    fn = {"pearson": pearsonr, "spearman": spearmanr, "kendall": kendalltau}[method]
    r, p = fn(pair["x"], pair["y"])
    return {"method": method, "r": float(r), "pvalue": float(p), "n": len(pair)}


def correlation_matrix(
    df: pd.DataFrame, columns: list[str] | None = None, method: str = "spearman", correction: str | None = "fdr_bh",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise correlation matrix plus a matching p-value matrix (corrected
    across all unique pairs unless `correction=None`).

    Returns:
        `(r_matrix, p_matrix)`, both features × features.
    """
    from itertools import combinations

    columns = columns or list(df.select_dtypes("number").columns)
    r = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    p = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns)
    pairs = list(combinations(columns, 2))
    raw = []
    for a, b in pairs:
        res = correlation(df[a], df[b], method=method)
        r.loc[a, b] = r.loc[b, a] = res["r"]
        raw.append(res["pvalue"])
    if pairs:
        adj = multipletests(np.nan_to_num(raw, nan=1.0), method=correction)[1] if correction else raw
        for (a, b), pv in zip(pairs, adj):
            p.loc[a, b] = p.loc[b, a] = pv
    return r, p


def linear_regression(df: pd.DataFrame, y_col: str, x_cols: list[str] | str, add_constant: bool = True):
    """Ordinary least squares (simple or multiple linear regression).

    Returns:
        The fitted statsmodels results — `.summary()`, `.params`,
        `.pvalues`, `.rsquared`, `.predict()` all work.
    """
    import statsmodels.api as sm

    x_cols = [x_cols] if isinstance(x_cols, str) else list(x_cols)
    sub = df[[y_col, *x_cols]].apply(pd.to_numeric, errors="coerce").dropna()
    X = sm.add_constant(sub[x_cols]) if add_constant else sub[x_cols]
    return sm.OLS(sub[y_col], X).fit()


def regression_table(model) -> pd.DataFrame:
    """Coefficient table (estimate, std error, t, p, 95% CI) for any fitted
    statsmodels regression."""
    ci = model.conf_int()
    return pd.DataFrame({
        "term": model.params.index, "estimate": model.params.values, "std_err": model.bse.values,
        "statistic": model.tvalues.values, "pvalue": model.pvalues.values,
        "ci_lower": ci.iloc[:, 0].values, "ci_upper": ci.iloc[:, 1].values,
    })


def mixed_model(df: pd.DataFrame, formula: str, group_col: str, re_formula: str | None = None):
    """Linear mixed-effects model (e.g. `formula="d13c ~ region"`,
    `group_col="batch"` for a random intercept per batch).

    Returns:
        The fitted statsmodels `MixedLMResults`.
    """
    import statsmodels.formula.api as smf

    return smf.mixedlm(formula, data=df, groups=df[group_col], re_formula=re_formula).fit()


def contingency_test(df: pd.DataFrame, row_col: str, col_col: str, test: str = "auto") -> dict:
    """Counts in categories (Result G): chi-square test, or Fisher's exact
    test when any expected cell count is below 5 (Fisher needs a 2×2 table;
    for larger sparse tables the chi-square result is returned with a
    warning flag)."""
    from scipy.stats import chi2_contingency, fisher_exact

    table = pd.crosstab(df[row_col], df[col_col])
    chi2, p_chi, dof, expected = chi2_contingency(table)
    sparse = bool((expected < 5).any())
    if test == "auto":
        test = "fisher" if sparse and table.shape == (2, 2) else "chi2"
    if test == "fisher":
        odds, p = fisher_exact(table.values)
        return {"test": "fisher", "statistic": odds, "pvalue": p, "table": table, "sparse": sparse}
    return {"test": "chi2", "statistic": chi2, "pvalue": p_chi, "dof": dof, "table": table,
            "sparse": sparse, "warning": "expected counts < 5" if sparse else None}

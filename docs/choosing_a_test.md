# Which statistical test should I use?

A decision guide for picking a statistical test for your data. You answer a short series of yes/no questions about your
**experimental design**, and the guide leads you to the right test.

---

## Contents

1. [How to use this guide](#1-how-to-use-this-guide)
2. [Normality and the automatic switch](#2-normality-and-the-automatic-switch)
3. [The questions](#3-the-questions)
4. [Result cards](#4-result-cards)
5. [Testing many compounds at once](#5-testing-many-compounds-at-once)
6. [Common pitfalls](#6-common-pitfalls)
7. [Quick reference table](#7-quick-reference-table)

---

## 1. How to use this guide

Start at [Question 1](#question-1--is-the-result-a-number-or-a-category) and
follow the YES/NO links until you reach a **🎯 Result**. That card in
[section 4](#4-result-cards) names the test, says what question it answers,
and says whether `isofood_center` supports it yet.

The questions are about **how the experiment was designed**: what was
compared, and whether the same samples were measured more than once. They
are not about the numbers you got. Ideally you can answer them before you
have any data.

> 🖼️ **SCHEMA PLACEHOLDER — S0: Master decision tree (whole guide on one page)**
>
> *To draw:* every question from section 3 as a diamond (YES / NO), every
> result as a rounded box, and one dashed box labelled
> "normality check — automatic" in front of each result.
>
> ![S0 – Master decision tree](images/s0_master_tree.png)

---

## 2. Normality and the automatic switch

Most of the tests in this guide come in **pairs**:

- **Parametric** (t-test, ANOVA, Pearson): these compare means and assume
  that the values in each group are roughly **normally distributed** (the
  bell curve). When that holds, they are the more powerful choice.
- **Non-parametric** (Mann-Whitney, Kruskal-Wallis, Spearman): these work on
  *ranks* (1st, 2nd, 3rd …) instead of raw values. They are robust to skewed
  data and outliers, but slightly less powerful when the data are actually
  normal.

> ⚠️ A *normal distribution* has nothing to do with *normalised* data.
> Normalisation (dividing by an internal standard, scaling to total area,
> etc.) is a preprocessing step. Normality is about the *shape* of the
> distribution of values.

The decision tree picks the **pair**. With the default `test="auto"`,
`isofood_center` then picks the test within the pair:

```
for each feature:
    run a Shapiro-Wilk normality test on every group
    if EVERY group has ≥ 3 values AND p > 0.01:
        → parametric test
    else:
        → non-parametric test
```

The `test` column in the output shows which test was run on each feature.
Report that test in your methods section.

> 🖼️ **SCHEMA PLACEHOLDER — S1: The automatic normality switch**
>
> *To draw:* input "measurement + groups" → diamond "Shapiro-Wilk: all groups
> normal AND n ≥ 3 in each?" → YES: "parametric test" / NO: "non-parametric
> test" → both arrows end in "result table (shows which test was used)".
>
> ![S1 – Automatic normality switch](images/s1_normality_switch.png)

**What is checked for each test pair:**

| Test pair | What the normality check looks at |
|---|---|
| Two or more independent groups | The values within **each group** |
| Paired, two measurements | The **within-sample differences** (after − before) |
| Paired, three or more measurements | The values at **each time point** |
| Correlation between two features | **Both** features |
| Categories (counts) | No normality check. The check is instead whether any expected cell count is below 5 (see [Result G](#result-g--counts-in-categories)) |

**Limits of the automatic switch.** Treat it as a sensible default, not a
guarantee:

- **Small *n* (below about 5–8 per group).** Shapiro-Wilk can barely detect
  non-normality, so clearly skewed data can still "pass". Every test has low
  power here, so a non-significant result does **not** show that there is no
  difference.
- **Large *n* (above about 50 per group).** Shapiro-Wilk flags even tiny,
  harmless departures from normality. Parametric tests cope well with these
  at large *n*.
- **Different tests for different features.** In one run, some features may
  get the parametric test and others the non-parametric one. That is fine,
  but report it. If you want one test for every feature (for example to
  match a published method), force it with `test="anova"` or
  `test="kruskal"`.
- **Concentration data.** Concentrations are often right-skewed. A log
  transform frequently makes them close to normal and is usually more
  interpretable than switching to a rank test. Do it before testing if it
  makes sense for your data.

---

## 3. The questions

### Question 1 — Is the result a number or a category?

A **number** is a measured value on a scale, such as δ13C = −25.4 ‰ or
Fe = 12.3 mg/kg. A **category** is a label you count, such as "pollen
present / absent" or "organic / conventional".

- **Number** → go to [Question 2](#question-2--are-you-comparing-groups)
- **Category** → 🎯 [Result G — Counts in categories](#result-g--counts-in-categories)

> 🖼️ **SCHEMA PLACEHOLDER — S2: Question 1 (number or category?)**
>
> ![S2 – Number or category](images/s2_q1_number_or_category.png)

---

### Question 2 — Are you comparing groups?

*YES:* "Does δ13C differ between coastal and mountain honey?"
*NO:* "When Fe increases, does Zn increase as well?" This asks about a
*relationship* between two features, not about groups.

- **YES** → go to [Question 3](#question-3--is-each-value-paired-with-a-value-from-the-same-sample)
- **NO** → go to [Question 7](#question-7--do-you-want-to-know-whether-two-features-vary-together)

---

### Question 3 — Is each value paired with a value from the same sample?

Answer **YES** if the same samples were measured at several time points,
before and after a treatment, under different conditions, or with two
different methods.

*YES:* the same 10 jars analysed fresh and after 6 months of storage.
*NO:* 10 jars from 2024 and 10 *different* jars from 2025.

*Test:* can you match every value in group A to exactly one value in group B
from the same physical sample? If you can, the data are paired.

> ⚠️ Treating paired data as independent wastes power: the test ignores the
> fact that each sample acts as its own control. Treating independent data
> as paired is simply wrong.

- **YES** → go to [Question 4](#question-4--are-there-exactly-two-measurements-per-sample)
- **NO** → go to [Question 5](#question-5--are-samples-classified-by-two-factors-at-once)

> 🖼️ **SCHEMA PLACEHOLDER — S3: Same samples or different samples?**
>
> *To draw:* left side, the same jar icon with arrows "t1 → t2" (paired);
> right side, two different sets of jars (independent).
>
> ![S3 – Same vs different samples](images/s3_q3_same_or_different.png)

---

### Question 4 — Are there exactly two measurements per sample?

*YES:* before / after storage.
*NO:* 0 / 3 / 6 / 12 months.

- **YES** → 🎯 [Result C — Paired, two measurements](#result-c--paired-two-measurements)
- **NO** → 🎯 [Result D — Paired, three or more measurements](#result-d--paired-three-or-more-measurements)

---

### Question 5 — Are samples classified by two factors at once?

*YES:* samples differ in **region** (coast / mountains) **and** in **season**
(spring / autumn). You want the effect of each factor, and to know whether
they interact (for example, the regional difference is larger in spring).
*NO:* samples differ by region only.

*Rule of thumb:* if the comparison needs **two** grouping columns (for
example `sample_group` *and* `sample_subgroup`), answer **YES**.

> 💡 If the second factor is only a nuisance (for example the analysis batch)
> and you don't care about its effect, this is still a two-factor design.
> Ignoring a factor that affects the values adds noise and can hide real
> differences.

- **YES** → 🎯 [Result E — Two grouping factors](#result-e--two-grouping-factors)
- **NO** → go to [Question 6](#question-6--are-there-exactly-two-groups)

---

### Question 6 — Are there exactly two groups?

The groups can be any categories: regions, producers, matrices, years.

*YES:* Slovenia vs. Croatia.
*NO:* Slovenia vs. Croatia vs. Italy.

> ⚠️ With three or more groups, do **not** run t-tests on every pair of
> groups. Each extra test adds a chance of a false positive. Use Result B,
> followed by a post-hoc test.

- **YES** → 🎯 [Result A — Two independent groups](#result-a--two-independent-groups)
- **NO** → 🎯 [Result B — Three or more independent groups](#result-b--three-or-more-independent-groups)

> 🖼️ **SCHEMA PLACEHOLDER — S4: Different samples branch (Questions 5 and 6)**
>
> ![S4 – Independent groups branch](images/s4_q5_q6_independent.png)

---

### Question 7 — Do you want to know whether two features vary together?

*YES:* "Do samples with more Fe also contain more Zn?"

- **YES** → 🎯 [Result F — Correlation between two features](#result-f--correlation-between-two-features)
- **NO** → this guide doesn't cover your question. Two common cases:
  - **One variable is something you set or know exactly** (time, depth,
    dose, temperature), and you want to describe *how much* the other
    changes per unit. That is **regression**, not correlation.
  - **You want to classify samples or find patterns** across the whole
    profile (for example, "can we tell a sample's origin from its VOC
    fingerprint?"). That is a multivariate question, not a single test. See
    the PCA / Random Forest / LDA tools in `isofood_center.pipelines.ml`.

> 🖼️ **SCHEMA PLACEHOLDER — S5: Relationship branch (Question 7)**
>
> ![S5 – Relationship branch](images/s5_q7_relationship.png)

---

## 4. Result cards

Each card shows a test pair. Unless you force a specific test, the
[automatic switch](#2-normality-and-the-automatic-switch) picks which one
is run.

Status: ✅ available in `isofood_center`

---

### Result A — Two independent groups

| | Parametric | Non-parametric |
|---|---|---|
| Test | Welch's t-test | Mann-Whitney U test |
| Status | ✅ | ✅ |

**Answers:** "Do the values differ between group 1 and group 2?"
**Example:** Does δ15N differ between Slovenian and Croatian honey?

Welch's t-test is preferred over the classic Student's t-test because it
doesn't assume that both groups have the same spread.

```python
stats.test_all_features(df_wide, ["d15n"], "origin")   # two groups → Welch / Mann-Whitney automatically
stats.welch_ttest(df_wide, "d15n", "origin")
stats.mann_whitney(df_wide, "d15n", "origin")
stats.effect_size(df_wide, "d15n", "origin")           # Cohen's d and fold change
```

---

### Result B — Three or more independent groups

| | Parametric | Non-parametric |
|---|---|---|
| Test | One-way ANOVA | Kruskal-Wallis test |
| Status | ✅ | ✅ |

**Answers:** "Does at least one group differ from the others?" It does
**not** tell you which groups differ.
**Example:** Does Fe differ between soils from the coast, the plains and the
mountains?

```python
from isofood_center.pipelines import stats

stats.test_all_features(df_wide, ["fe", "zn"], "sample_group")
```

**Next step:** if the result is significant and you need to know *which*
groups differ, run a post-hoc test. The same switch chooses it: Tukey HSD
(parametric) or Dunn's test (non-parametric). Status: ✅

```python
stats.posthoc(df_wide, "fe", "sample_group")   # Tukey HSD or Dunn, picked by the normality check
stats.effect_size(df_wide, "fe", "sample_group")  # eta²
stats.levene_test(df_wide, "fe", "sample_group")  # equal variances?
```

> 🖼️ **SCHEMA PLACEHOLDER — S6: Post-hoc step after ANOVA / Kruskal-Wallis**
>
> ![S6 – Post-hoc step](images/s6_posthoc.png)

---

### Result C — Paired, two measurements

| | Parametric | Non-parametric |
|---|---|---|
| Test | Paired t-test | Wilcoxon signed-rank test |
| Status | ✅ | ✅ |

**Answers:** "Did the values change between the two measurements?"
**Example:** Did the hexanal peak area in the same jars change after
6 months of storage?

> ⚠️ Only samples measured at **both** time points are used. Samples with
> only one measurement are dropped.

```python
# long format: one row per sample × time point
stats.paired_test(df, "hexanal", subject_col="sample_id", condition_col="storage")
```

---

### Result D — Paired, three or more measurements

| | Parametric | Non-parametric |
|---|---|---|
| Test | Repeated-measures ANOVA | Friedman test |
| Status | ✅ | ✅ |

**Answers:** "Do the values change across the time points?"
**Example:** Does δ13C in the same samples change across 0, 3, 6 and
12 months?

As with Result B, a significant result says only that *some* time points
differ. A post-hoc test is needed to find which ones.

```python
stats.repeated_measures_test(df, "d13c", subject_col="sample_id", condition_col="month")
```

---

### Result E — Two grouping factors

| | Parametric | Non-parametric |
|---|---|---|
| Test | Two-way ANOVA | Aligned Rank Transform (ART) ANOVA |
| Status | ✅ | ✅ |

```python
stats.two_way_anova(df_wide, "zn", "region", "year")   # switch decided by normality of the residuals
viz.interaction_plot(df_wide, "zn", "region", "year")
```

**Answers:** "Does factor 1 matter? Does factor 2 matter? Does the effect of
factor 1 depend on factor 2 (interaction)?"
**Example:** Is Zn affected by region, by year, and by the combination of
the two?

> 💡 Results are hardest to interpret when the interaction is significant.
> The effect of one factor then can't be described without naming the
> level of the other. Plot the group means before you write anything up,
> and ask a colleague with stats experience to check unbalanced designs
> (unequal numbers of samples per combination).

---

### Result F — Correlation between two features

| | Parametric | Non-parametric |
|---|---|---|
| Test | Pearson correlation | Spearman correlation |
| Status | ✅ | ✅ |

```python
stats.correlation(df_wide["fe"], df_wide["zn"])          # Pearson or Spearman, auto
r, p = stats.correlation_matrix(df_wide, ["fe", "zn", "mn"], method="spearman")
viz.correlation_heatmap(r, p)
model = stats.linear_regression(df_wide, "zn", ["fe", "mn"])   # (multiple) linear regression
stats.regression_table(model)
```

**Answers:** "When one feature goes up, does the other tend to go up (or
down)?"
**Example:** Do samples with more Fe also contain more Zn?

- Pearson measures a **linear** relationship. Spearman measures any
  **monotonic** one (always increasing or always decreasing, not
  necessarily in a straight line).
- **Always plot the data.** A single outlier can create or hide a
  correlation, and a strong curved relationship can give r ≈ 0.

> ⚠️ Correlation does not imply causation. Two elements can correlate
> simply because both depend on a third factor, such as soil type or the
> analysis batch.

---

### Result G — Counts in categories

| | Enough data (all expected counts ≥ 5) | Sparse data (any expected count < 5) |
|---|---|---|
| Test | Chi-square test | Fisher's exact test |
| Status | ✅ | ✅ |

```python
stats.contingency_test(df, "origin", "pollen_marker")
```

Here the switch is not about normality. It checks whether each cell of the
contingency table is expected to hold enough samples for the chi-square
approximation to be reliable.

**Answers:** "Does the proportion of a category differ between groups?"
**Example:** Is "pollen present" more common in coastal than in mountain
honey?

> 💡 Values below the limit of detection (LOD) are **not** a category
> question if you have the measured values. Turning concentrations into
> "detected / not detected" throws away information. Only use this branch
> if presence/absence is all you have, or all you care about.

> 🖼️ **SCHEMA PLACEHOLDER — S7: Category branch (chi-square vs. Fisher switch)**
>
> ![S7 – Category branch](images/s7_categories.png)

---

## 5. Testing many compounds at once

A typical dataset has many features: dozens of elements, or hundreds of VOCs.
Every test has an α chance of a false positive (`isofood_center` uses α = 0.01 by
default). If you run 100 tests at α = 0.01 and none of the compounds truly
differs, you should still expect about one "significant" result by chance
alone.

`test_all_features` therefore **corrects the p-values** across all tested
features. To decide what is significant, use `pvalue_corrected` (and the
`significant` column, which is based on it), not the raw `pvalue`.

| Correction | What it controls | When to use | In `isofood_center` |
|---|---|---|---|
| Benjamini-Hochberg (FDR) | The expected *fraction* of false positives among the hits | Screening many compounds for candidates | ✅ default (`correction="fdr_bh"`) |
| Bonferroni | The chance of *any* false positive at all | Few features, or when one false positive is costly | `correction="bonferroni"` |

> 💡 At FDR 0.01, about 1 in 100 of your significant compounds is expected to
> be a false positive. That is usually acceptable for screening. Confirm
> important findings in an independent sample set.

> 🖼️ **SCHEMA PLACEHOLDER — S8: From many p-values to a list of significant compounds**
>
> ![S8 – Multiple testing correction](images/s8_multiple_testing.png)

---

## 6. Common pitfalls

- **Counting technical replicates as samples.** Triplicate injections of one
  extract are one sample, not three. Average them first (or model them
  properly). Otherwise *n* is inflated and p-values are far too small.
- **Treating "not significant" as "no difference".** With small groups,
  absence of evidence is not evidence of absence. Report effect sizes (mean
  difference, fold change) alongside the p-value.
- **Treating "significant" as "important".** With many samples, tiny and
  practically irrelevant differences become significant. Check whether the
  size of the difference matters chemically, and whether it is larger than
  your method's measurement uncertainty.
- **Testing first, deciding later.** Choosing the test, the groups or which
  outliers to remove *after* looking at the p-values makes them invalid.
  Decide on these from the experimental design.
- **Hidden batch effects.** If all samples from one region were analysed on
  one day and all from another region on a different day, a "regional"
  difference may just be instrument drift. Randomise the run order where
  possible.

---

## 7. Quick reference table

| Path through the questions | Result | Parametric | Non-parametric |
|---|---|---|---|
| Q1 category | G | Chi-square* | Fisher's exact* |
| Q1 number → Q2 YES → Q3 NO → Q5 NO → Q6 YES | A | Welch's t-test | Mann-Whitney U |
| Q1 number → Q2 YES → Q3 NO → Q5 NO → Q6 NO | B | One-way ANOVA | Kruskal-Wallis |
| Q1 number → Q2 YES → Q3 NO → Q5 YES | E | Two-way ANOVA | ART ANOVA |
| Q1 number → Q2 YES → Q3 YES → Q4 YES | C | Paired t-test | Wilcoxon signed-rank |
| Q1 number → Q2 YES → Q3 YES → Q4 NO | D | Repeated-measures ANOVA | Friedman |
| Q1 number → Q2 NO → Q7 YES | F | Pearson | Spearman |

\* For Result G the two columns mean "enough data" and "sparse data", not
parametric and non-parametric.

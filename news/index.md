# Changelog

## DPI 2026.2

- Minor changes and improvements.

## DPI 2025.11

CRAN release: 2025-11-24

- Improved
  [`DPI_dag()`](https://psychbruce.github.io/DPI/reference/DPI_dag.md)
  and
  [`plot.dpi.dag()`](https://psychbruce.github.io/DPI/reference/S3method.network.md).
  Also forced the plot to strictly show a DAG while changing edge color
  of insignificant DPI into faded grey.
- (Fixed in patch version 2025.10-1) Fixed
  [`DPI_curve()`](https://psychbruce.github.io/DPI/reference/DPI_curve.md)
  for wrong (reverse) direction of DPI caused by the change of parameter
  order of `x` and `y` in version 2025.10.
- (Fixed in patch version 2025.10-1) Fixed a bug caused by `dpi`
  parameter-object name conflict (internally) when saving
  [`DPI()`](https://psychbruce.github.io/DPI/reference/DPI.md) results
  into a `file`.

## DPI 2025.10

CRAN release: 2025-10-16

**This version contains breaking changes to function names and
visualization methods.**

- Added
  [`DPI_dag()`](https://psychbruce.github.io/DPI/reference/DPI_dag.md):
  Directed acyclic graphs (DAGs) via DPI exploratory analysis (causal
  discovery) for all significant partial correlations.
- Added `bonf` and `pseudoBF` parameters to
  [`DPI()`](https://psychbruce.github.io/DPI/reference/DPI.md),
  [`DPI_curve()`](https://psychbruce.github.io/DPI/reference/DPI_curve.md),
  and
  [`DPI_dag()`](https://psychbruce.github.io/DPI/reference/DPI_dag.md).
  - `bonf`: Bonferroni correction to control for false positive rates
    among multiple pairwise DPI tests.
  - `pseudoBF`: Use normalized pseudo Bayes Factors
    `sigmoid(log(PseudoBF10))` as the Significance score (0~1). Pseudo
    Bayes Factors are computed using the transformation rules proposed
    by Wagenmakers (2022) <https://doi.org/10.31234/osf.io/egydq>.
- Added S3 methods
  [`plot.cor.net()`](https://psychbruce.github.io/DPI/reference/S3method.network.md),
  [`plot.bns.dag()`](https://psychbruce.github.io/DPI/reference/S3method.network.md),
  and
  [`plot.dpi.dag()`](https://psychbruce.github.io/DPI/reference/S3method.network.md)
  that can transform `qgraph` base-plot objects into `ggplot` objects
  for more stable and flexible visualization.
- Added
  [`p_to_bf()`](https://psychbruce.github.io/DPI/reference/p_to_bf.md):
  Convert *p* values to pseudo Bayes Factors ($\text{PseudoBF}_{10}$).
- Renamed `cor_network()` to
  [`cor_net()`](https://psychbruce.github.io/DPI/reference/cor_net.md),
  `dag_network()` to
  [`BNs_dag()`](https://psychbruce.github.io/DPI/reference/BNs_dag.md),
  and `matrix_cor()` to
  [`cor_matrix()`](https://psychbruce.github.io/DPI/reference/cor_matrix.md).
- Fixed
  [`cor_net()`](https://psychbruce.github.io/DPI/reference/cor_net.md)
  to return the exactly correct *p* values of (partial) correlation
  coefficients.
- Improved output information in console and plot.

## DPI 2025.9

CRAN release: 2025-09-20

**This version contains breaking changes to both algorithm and
functionality.**

- Refined [`DPI()`](https://psychbruce.github.io/DPI/reference/DPI.md)
  algorithm to limit $\text{DPI} \in ( - 1,1)$ and also simplified its
  output information. $$\begin{aligned}
  \text{DPI}_{X\rightarrow Y} & {= \text{Direction}_{X\rightarrow Y} \cdot \text{Significance}_{X\rightarrow Y}} \\
   & {= \text{Delta}\left( R^{2} \right) \cdot \text{Sigmoid}\left( \frac{p}{\alpha} \right)} \\
   & {= \left( R_{Y \sim X + Covs}^{2} - R_{X \sim Y + Covs}^{2} \right) \cdot \left( 1 - \tanh\frac{p_{XY|Covs}}{2\alpha} \right)} \\
   & {\in ( - 1,1)}
  \end{aligned}$$
  - In an earlier version of algorithm, the strength score was computed
    as
    $t_{\beta_{XY|Covs}}^{2} = t_{r.partial_{XY|Covs}}^{2} \in \lbrack 0, + \infty)$.
    While this algorithm performs as well as the new
    $\text{Sigmoid}\left( \frac{p}{\alpha} \right)$ approach (e.g., both
    have low false positive and false negative rates), $t^{2}$ has a
    major flaw that its values cannot converge to a limited range so
    that the final DPI values would be heavily determined by $t^{2}$,
    which is not a desired attribute. In contrast, the new algorithm can
    make the significance score more likely to be an “on-off switch”,
    with values more likely approximating 0 or 1, thereby minimizing its
    impact on the interpretation of final DPI values.
- Renamed `data_random()` to
  [`sim_data()`](https://psychbruce.github.io/DPI/reference/sim_data.md)
  with enhanced functionality that supports data simulation from a
  multivariate normal distribution, using
  [`MASS::mvrnorm()`](https://rdrr.io/pkg/MASS/man/mvrnorm.html).
- Added
  [`sim_data_exp()`](https://psychbruce.github.io/DPI/reference/sim_data_exp.md):
  Simulate experiment-like data with *independent* binary Xs.
- Used [`gc()`](https://rdrr.io/r/base/gc.html) in
  [`DPI()`](https://psychbruce.github.io/DPI/reference/DPI.md),
  [`DPI_curve()`](https://psychbruce.github.io/DPI/reference/DPI_curve.md),
  and `dag_network()` for memory garbage collection.
- Provided a better example in `dag_network()` for arranging multiple
  base-R-style plots using
  [`aplot::plot_list()`](https://rdrr.io/pkg/aplot/man/plot_list.html).

## DPI 2025.8

CRAN release: 2025-08-20

- Added `dag_network()`: Directed acyclic graphs (DAGs) via causal
  Bayesian networks (BNs).
- Improved `cor_network()`: Correlation and partial correlation
  networks.
- Moved help pages of all S3 method functions to `S3method.dpi` and
  `S3method.network` and made them as internal topics.

## DPI 2025.6

CRAN release: 2025-06-18

- CRAN package publication.
- Initial public release on [GitHub](https://github.com/psychbruce/DPI).
- Developed core functions and package logo.

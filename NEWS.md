**Please check the [latest news (change log)](https://psychbruce.github.io/DPI/news/index.html) and keep this package updated.**

# DPI 2025.11

-   Improved `DPI_dag()` and `plot.dpi.dag()`.
-   (Fixed in patch version 2025.10-1) Fixed `DPI_curve()` for wrong (reverse) direction of DPI caused by the change of parameter order of `x` and `y` in version 2025.10.
-   (Fixed in patch version 2025.10-1) Fixed a bug caused by `dpi` parameter-object name conflict (internally) when saving `DPI()` results into a `file`.

# DPI 2025.10

**This version contains breaking changes to function names and visualization methods.**

-   Added `DPI_dag()`: Directed acyclic graphs (DAGs) via DPI exploratory analysis (causal discovery) for all significant partial correlations.
-   Added `bonf` and `pseudoBF` parameters to `DPI()`, `DPI_curve()`, and `DPI_dag()`.
    -   `bonf`: Bonferroni correction to control for false positive rates among multiple pairwise DPI tests.
    -   `pseudoBF`: Use normalized pseudo Bayes Factors `sigmoid(log(PseudoBF10))` as the Significance score (0\~1). Pseudo Bayes Factors are computed using the transformation rules proposed by Wagenmakers (2022) <https://doi.org/10.31234/osf.io/egydq>.
-   Added S3 methods `plot.cor.net()`, `plot.bns.dag()`, and `plot.dpi.dag()` that can transform `qgraph` base-plot objects into `ggplot` objects for more stable and flexible visualization.
-   Added `p_to_bf()`: Convert *p* values to pseudo Bayes Factors ($\text{PseudoBF}_{10}$).
-   Renamed `cor_network()` to `cor_net()`, `dag_network()` to `BNs_dag()`, and `matrix_cor()` to `cor_matrix()`.
-   Fixed `cor_net()` to return the exactly correct *p* values of (partial) correlation coefficients.
-   Improved output information in console and plot.

# DPI 2025.9

**This version contains breaking changes to both algorithm and functionality.**

-   Refined `DPI()` algorithm to limit $\text{DPI} \in (-1, 1)$ and also simplified its output information. $$
    \begin{aligned}
    \text{DPI}_{X \rightarrow Y}
    & = \text{Direction}_{X \rightarrow Y} \cdot \text{Significance}_{X \rightarrow Y} \\
    & = \text{Delta}(R^2) \cdot \text{Sigmoid}(\frac{p}{\alpha}) \\
    & = \left( R_{Y \sim X + Covs}^2 - R_{X \sim Y + Covs}^2 \right) \cdot \left( 1 - \tanh \frac{p_{XY|Covs}}{2\alpha} \right) \\
    & \in (-1, 1)
    \end{aligned}
    $$
    -   In an earlier version of algorithm, the strength score was computed as $t_{\beta_{XY|Covs}}^2 = t_{r.partial_{XY|Covs}}^2 \in [0, +\infty)$. While this algorithm performs as well as the new $\text{Sigmoid}(\frac{p}{\alpha})$ approach (e.g., both have low false positive and false negative rates), $t^2$ has a major flaw that its values cannot converge to a limited range so that the final DPI values would be heavily determined by $t^2$, which is not a desired attribute. In contrast, the new algorithm can make the significance score more likely to be an "on-off switch", with values more likely approximating 0 or 1, thereby minimizing its impact on the interpretation of final DPI values.
-   Renamed `data_random()` to `sim_data()` with enhanced functionality that supports data simulation from a multivariate normal distribution, using `MASS::mvrnorm()`.
-   Added `sim_data_exp()`: Simulate experiment-like data with *independent* binary Xs.
-   Used `gc()` in `DPI()`, `DPI_curve()`, and `dag_network()` for memory garbage collection.
-   Provided a better example in `dag_network()` for arranging multiple base-R-style plots using `aplot::plot_list()`.

# DPI 2025.8

-   Added `dag_network()`: Directed acyclic graphs (DAGs) via causal Bayesian networks (BNs).
-   Improved `cor_network()`: Correlation and partial correlation networks.
-   Moved help pages of all S3 method functions to `S3method.dpi` and `S3method.network` and made them as internal topics.

# DPI 2025.6

-   CRAN package publication.
-   Initial public release on [GitHub](https://github.com/psychbruce/DPI).
-   Developed core functions and package logo.

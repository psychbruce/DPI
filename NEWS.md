**Please check the [latest news (change log)](https://psychbruce.github.io/DPI/news/index.html) and keep this package updated.**

# DPI 2025.10

**This version contains breaking changes to function names.**

-   Added `DPI_dag()`: Directed acyclic graphs (DAGs) via the DPI exploratory analysis for all significant partial correlations.
-   Fixed `cor_network()` to return the correct *p* values of (partial) correlation coefficients.
-   Renamed `cor_network()` to `cor_net()`, `dag_network()` to `BNs_dag()`, and `matrix_cor()` to `cor_matrix()`.

# DPI 2025.9

**This version contains breaking changes to both algorithm and functionality.**

-   Refined `DPI()` algorithm to limit $\text{DPI} \in (-1, 1)$ and also simplified its output information. $$
    \begin{aligned}
    \text{DPI}_{X \rightarrow Y}
    & = \text{Direction}_{X \rightarrow Y} \cdot \text{Strength}_{XY} \\
    & = \text{Delta}(R^2) \cdot \text{Sigmoid}(\frac{p}{\alpha}) \\
    & = \left( R_{Y \sim X + Covs}^2 - R_{X \sim Y + Covs}^2 \right) \cdot \left( 1 - \tanh \frac{p_{XY|Covs}}{2\alpha} \right) \\
    & \in (-1, 1)
    \end{aligned}
    $$
    -   In an earlier version of algorithm, the strength score was computed as $t_{\beta_{XY|Covs}}^2 = t_{r.partial_{XY|Covs}}^2 \in [0, +\infty)$. While this algorithm performs as well as the new $\text{Sigmoid}(\frac{p}{\alpha})$ approach (e.g., with low false positive and false negative rates), $t^2$ has a major flaw that its values cannot converge to a limited range so that the final DPI values would be heavily determined by $t^2$, which is not a desired attribute. In contrast, the new algorithm can make the strength score more likely to be an on-off switch, with values approximating 0 or 1, thereby minimizing its impact on the interpretation of final DPI values.
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

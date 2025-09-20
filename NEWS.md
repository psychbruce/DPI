**Please check the [latest news (change log)](https://psychbruce.github.io/DPI/news/index.html) and keep this package updated.**

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

# DPI <img src="man/figures/logo.png" align="right" height="160"/>

🛸 The Directed Prediction Index (DPI).

The *Directed Prediction Index* (DPI) is a quasi-causal inference method for cross-sectional data designed to quantify the *relative endogeneity* (relative dependence) of outcome (*Y*) versus predictor (*X*) variables in regression models.

<!-- badges: start -->

[![CRAN-Version](https://www.r-pkg.org/badges/version/DPI?color=red)](https://CRAN.R-project.org/package=DPI) [![GitHub-Version](https://img.shields.io/github/r-package/v/psychbruce/DPI?label=GitHub&color=orange)](https://github.com/psychbruce/DPI) [![R-CMD-check](https://github.com/psychbruce/DPI/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/psychbruce/DPI/actions/workflows/R-CMD-check.yaml) [![CRAN-Downloads](https://cranlogs.r-pkg.org/badges/grand-total/DPI)](https://CRAN.R-project.org/package=DPI) [![GitHub-Stars](https://img.shields.io/github/stars/psychbruce/DPI?style=social)](https://github.com/psychbruce/DPI/stargazers)

<!-- badges: end -->

<img src="https://psychbruce.github.io/img/CC-BY-NC-SA.jpg" width="120px" height="42px"/>

## Author

Bruce H. W. S. Bao 包寒吴霜

📬 [baohws\@foxmail.com](mailto:baohws@foxmail.com)

📋 [psychbruce.github.io](https://psychbruce.github.io)

## Citation

-   Bao, H. W. S. (2025). *DPI: The Directed Prediction Index for quasi-causal inference with cross-sectional data*. <https://doi.org/10.32614/CRAN.package.DPI>
-   Bao, H. W. S. (Manuscript in preparation). *The Directed Prediction Index (DPI): Quasi-causal inference for observational data from relative endogeneity of variables*.

## Installation

``` r
## Method 1: Install from CRAN
install.packages("DPI")

## Method 2: Install from GitHub
install.packages("devtools")
devtools::install_github("psychbruce/DPI", force=TRUE)
```

## Algorithm Details

Define $\text{DPI}$ as the product of $\text{Direction}$ (relative direction) and $\text{Strength}$ (absolute strength) of the expected $X \rightarrow Y$ relationship:

$$
\begin{aligned}
\text{DPI}_{X \rightarrow Y}
& = \text{Direction}_{X \rightarrow Y} \cdot \text{Strength}_{XY} \\
& = \text{Delta}(R^2) \cdot \text{Sigmoid}(\frac{p}{\alpha}) \\
& = \left( R_{Y \sim X + Covs}^2 - R_{X \sim Y + Covs}^2 \right) \cdot \left( 1 - \tanh \frac{p_{XY|Covs}}{2\alpha} \right) \\
& \in (-1, 1)
\end{aligned}
$$

In econometrics and broader social sciences, an *exogenous* variable is assumed to have a directed (causal or quasi-causal) influence on an *endogenous* variable ($ExoVar \rightarrow EndoVar$). By quantifying the *relative endogeneity* of outcome versus predictor variables in multiple linear regression models, the DPI can suggest a plausible (admissible) direction of influence (i.e., $\text{DPI}_{X \rightarrow Y} > 0 \text{: } X \rightarrow Y$) after controlling for a sufficient number of possible confounders and simulated random covariates.

### Key Steps of Conceptualization and Computation

All steps have been compiled into the functions `DPI()` and `DPI_curve()`. See their help pages for usage and illustrative examples. Below are conceptual rationales and mathematical explanations.

#### Step 1: Relative Direction

Define $\text{Direction}_{X \rightarrow Y}$ as *relative endogeneity* (relative dependence) of $Y$ vs. $X$ in a given variable set involving all possible confounders $Covs$:

$$
\begin{aligned}
\text{Direction}_{X \rightarrow Y} & = \text{Endogeneity}(Y) - \text{Endogeneity}(X) \\
& = R_{Y \sim X + Covs}^2 - R_{X \sim Y + Covs}^2 \\
& = \text{Delta}(R^2) \\
& \in (-1, 1)
\end{aligned}
$$

It uses $\text{Delta}(R^2)$ to test whether $Y$ (outcome), compared to $X$ (predictor), can be *more strongly predicted* by all $m$ observable control variables (included in a given sample) and $k$ unobservable random covariates (randomly generated in simulation samples, as specified by `k.cov` in the `DPI()` function). A higher $R^2$ indicates *higher dependence* (i.e., *higher endogeneity*) in a given variable set.

#### Step 2: Absolute Strength

Define $\text{Sigmoid}(\frac{p}{\alpha})$ as *absolute strength* of the partial relationship between $X$ and $Y$ when controlling for all possible confounders $Covs$:

$$
\begin{aligned}
\text{Sigmoid}(\frac{p}{\alpha}) & = 2 \left[ 1 - \text{sigmoid}(\frac{p_{XY|Covs}}{\alpha}) \right] \\
& = 1 - \tanh \frac{p_{XY|Covs}}{2\alpha} \\
& \in (0, 1)
\end{aligned}
$$

It uses $\text{Sigmoid}(\frac{p}{\alpha})$ to penalize insignificant ($p > \alpha$) partial relationship between $X$ and $Y$. Partial correlation $r_{partial}$ always has the equivalent $t$ test and the same $p$ value as partial regression coefficient $\beta_{partial}$ between $Y$ and $X$. A higher $\text{Sigmoid}(\frac{p}{\alpha})$ indicates a more likely (less spurious) partial relationship when controlling for all possible confounders.

Notes on transformation among $\tanh(x)$, $\text{sigmoid}(x)$, and $\text{Sigmoid}(\frac{p}{\alpha})$:

$$
\begin{aligned}
\tanh(x) & = \frac{e^x - e^{-x}}{e^x + e^{-x}} \\
& = 1 - \frac{2}{1 + e^{2x}} \\
& = \frac{2}{1 + e^{-2x}} - 1 \\
& = 2 \cdot \text{sigmoid}(2x) - 1, & \in (-1, 1) \\
\text{sigmoid}(x) & = \frac{1}{1 + e^{-x}} \\
& = \frac{\tanh(\frac{x}{2}) + 1}{2}, & \in (0, 1) \\
\text{Sigmoid}(\frac{p}{\alpha}) & = 2 \left[ 1 - \text{sigmoid}(\frac{p}{\alpha}) \right] \\
& = 1 - \tanh \frac{p}{2\alpha}. & \in (0, 1)
\end{aligned}
$$

| $p$ | $\text{Sigmoid}(\frac{p}{\alpha})$ with $\alpha = 0.05$ |
|--------------------------|----------------------------------------------|
| (\~0) | (\~1) |
| 0.0001 | 0.999 |
| 0.001 | 0.990 |
| 0.01 | 0.900 |
| 0.02 | 0.803 |
| 0.03 | 0.709 |
| 0.04 | 0.620 |
| 0.05 ($\frac{p}{\alpha}$ = 1) | 0.538 |
| 0.10 | 0.238 |
| 0.20 | 0.036 |
| 0.50 | 0.00009 |
| 0.80 | 0.0000002 |
| 1 | 0.000000004 |

#### Step 3: Data Simulation

**(1) Main analysis using `DPI()`**: Simulate `n.sim` random samples, with `k.cov` (unobservable) random covariate(s) in each simulated sample, to test the statistical significance of DPI.

**(2) Robustness check using `DPI_curve()`**: Run a series of DPI simulation analyses respectively with `1`\~`k.covs` (usually 1\~10) random covariates, producing a curve of DPIs (estimates, 95% CI, and 99% CI; usually getting closer to 0 as `k.covs` increases) that can indicate its sensitivity in identifying the directed prediction (i.e., *How many random covariates can DPIs survive to remain significant?*).

## Other Functions

This package also includes other functions helpful for exploring variable relationships and performing simulation studies.

-   Network analysis functions

    -   `cor_network()`: Correlation and partial correlation networks.

    -   `dag_network()`: Directed acyclic graphs (DAGs) via Bayesian networks (BNs).

-   Data simulation utility functions

    -   `matrix_cor()`: Produce a symmetric correlation matrix from values.

    -   `sim_data()`: Simulate data from a multivariate normal distribution.

    -   `sim_data_exp()`: Simulate experiment-like data with *independent* binary Xs.

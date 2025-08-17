# DPI <img src="man/figures/logo.png" align="right" height="160"/>

🛸 The Directed Prediction Index (DPI).

The *Directed Prediction Index* (DPI) is a simulation-based method for quantifying the *relative endogeneity* (relative dependence) of outcome (*Y*) versus predictor (*X*) variables in multiple linear regression models.

<!-- badges: start -->

[![CRAN-Version](https://www.r-pkg.org/badges/version/DPI?color=red)](https://CRAN.R-project.org/package=DPI) [![GitHub-Version](https://img.shields.io/github/r-package/v/psychbruce/DPI?label=GitHub&color=orange)](https://github.com/psychbruce/DPI) [![R-CMD-check](https://github.com/psychbruce/DPI/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/psychbruce/DPI/actions/workflows/R-CMD-check.yaml) [![CRAN-Downloads](https://cranlogs.r-pkg.org/badges/grand-total/DPI)](https://CRAN.R-project.org/package=DPI) [![GitHub-Stars](https://img.shields.io/github/stars/psychbruce/DPI?style=social)](https://github.com/psychbruce/DPI/stargazers)

<!-- badges: end -->

<img src="https://psychbruce.github.io/img/CC-BY-NC-SA.jpg" width="120px" height="42px"/>

## Author

Bruce H. W. S. Bao 包寒吴霜

📬 [baohws\@foxmail.com](mailto:baohws@foxmail.com)

📋 [psychbruce.github.io](https://psychbruce.github.io)

## Citation

-   Bao, H. W. S. (2025). *DPI: The Directed Prediction Index*. <https://doi.org/10.32614/CRAN.package.DPI>

## Installation

``` r
## Method 1: Install from CRAN
install.packages("DPI")

## Method 2: Install from GitHub
install.packages("devtools")
devtools::install_github("psychbruce/DPI", force=TRUE)
```

## Computation Details

$$
\begin{aligned}
\text{DPI}_{X \rightarrow Y} & = t^2 \cdot \Delta R^2 \\
& = t_{\beta_{XY|Covs}}^2 \cdot (R_{Y \sim X + Covs}^2 - R_{X \sim Y + Covs}^2) \\
& = t_{partial.r_{XY|Covs}}^2 \cdot (R_{Y \sim X + Covs}^2 - R_{X \sim Y + Covs}^2)
\end{aligned}
$$

In econometrics and broader social sciences, an *exogenous* variable is assumed to have a unidirectional (causal or quasi-causal) influence on an *endogenous* variable ($ExoVar \rightarrow EndoVar$). By quantifying the *relative endogeneity* of outcome versus predictor variables in multiple linear regression models, the DPI can suggest a more plausible direction of influence (e.g., $\text{DPI}_{X \rightarrow Y} > 0 \text{: } X \rightarrow Y$) after controlling for a sufficient number of potential confounding variables.

1.  It uses $\Delta R_{Y vs. X}^2$ to test whether $Y$ (outcome), compared to $X$ (predictor), can be *more strongly predicted* by $m$ observable control variables (included in a regression model) and $k$ unobservable random covariates (specified by `k.cov`; see the `DPI()` function). A higher $R^2$ indicates *relatively higher dependence* (i.e., *relatively higher endogeneity*) in a given variable set.
2.  It also uses $t_{partial.r}^2$ to penalize insignificant partial correlation ($r_{partial}$, with equivalent $t$ test as $\beta_{partial}$) between $Y$ and $X$, while ignoring the sign ($\pm$) of this correlation. A higher $t^2$ (equivalent to $F$ test value when $df = 1$) indicates a more robust (less spurious) partial relationship when controlling for other variables.
3.  Simulation samples with `k.cov` random covariates are generated to test the statistical significance of DPI.

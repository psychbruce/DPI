# DPI <img src="man/figures/logo.png" align="right" height="160"/>

🛸 The Directed Prediction Index (DPI).

The *Directed Prediction Index* (DPI) is a simulation-based method for quantifying *relative endogeneity* of outcome versus predictor variables in multiple linear regression models.

Computation of DPI with simulation samples:

$$
\begin{aligned}
\text{DPI}_{X \rightarrow Y} & = t^2 \cdot \Delta R^2 \\
& = t_{\beta_{XY|Covs}}^2 \cdot (R_{Y \sim X + Covs}^2 - R_{X \sim Y + Covs}^2) \\
& = t_{partial.r_{XY|Covs}}^2 \cdot (R_{Y \sim X + Covs}^2 - R_{X \sim Y + Covs}^2)
\end{aligned}
$$

<!-- badges: start -->

[![CRAN-Version](https://www.r-pkg.org/badges/version/DPI?color=red)](https://CRAN.R-project.org/package=DPI) [![GitHub-Version](https://img.shields.io/github/r-package/v/psychbruce/DPI?label=GitHub&color=orange)](https://github.com/psychbruce/DPI) [![R-CMD-check](https://github.com/psychbruce/DPI/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/psychbruce/DPI/actions/workflows/R-CMD-check.yaml) [![CRAN-Downloads](https://cranlogs.r-pkg.org/badges/grand-total/DPI)](https://CRAN.R-project.org/package=DPI) [![GitHub-Stars](https://img.shields.io/github/stars/psychbruce/DPI?style=social)](https://github.com/psychbruce/DPI/stargazers)

<!-- badges: end -->

<img src="https://psychbruce.github.io/img/CC-BY-NC-SA.jpg" width="120px" height="42px"/>

## Author

Han-Wu-Shuang (Bruce) Bao 包寒吴霜

📬 [baohws\@foxmail.com](mailto:baohws@foxmail.com)

📋 [psychbruce.github.io](https://psychbruce.github.io)

## Citation

-   Bao, H.-W.-S. (2025). *DPI: The Directed Prediction Index*. <https://CRAN.R-project.org/package=DPI>
    -   *Note*: This is the original citation. Please refer to the information when you `library(DPI)` for the APA-7 format of the version you installed.

## Installation

``` r
## Method 1: Install from CRAN
install.packages("DPI")

## Method 2: Install from GitHub
install.packages("devtools")
devtools::install_github("psychbruce/DPI", force=TRUE)
```

# DPI: The Directed Prediction Index for Causal Direction Inference from Observational Data

The Directed Prediction Index ('DPI') is a causal discovery method for
observational data designed to quantify the relative endogeneity of
outcome (Y) versus predictor (X) variables in regression models. By
comparing the coefficients of determination (R-squared) between the
Y-as-outcome and X-as-outcome models while controlling for sufficient
confounders and simulating k random covariates, it can quantify relative
endogeneity, providing a necessary but insufficient condition for causal
direction from a less endogenous variable (X) to a more endogenous
variable (Y). Methodological details are provided at
<https://psychbruce.github.io/DPI/>. This package also includes
functions for data simulation and network analysis (correlation, partial
correlation, and Bayesian Networks).

## See also

Useful links:

- <https://psychbruce.github.io/DPI/>

- Report bugs at <https://github.com/psychbruce/DPI/issues>

## Author

**Maintainer**: Han Wu Shuang Bao <baohws@foxmail.com>
([ORCID](https://orcid.org/0000-0003-3043-710X))

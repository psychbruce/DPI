# DPI: The Directed Prediction Index for Causal Direction Inference from Observational Data

The Directed Prediction Index ('DPI') is a quasi-causal inference
(causal discovery) method for observational data designed to quantify
the relative endogeneity (relative dependence) of outcome (Y) versus
predictor (X) variables in regression models. By comparing the
proportion of variance explained (R-squared) between the Y-as-outcome
model and the X-as-outcome model while controlling for a sufficient
number of possible confounders, it can suggest a plausible (admissible)
direction of influence from a less endogenous variable (X) to a more
endogenous variable (Y). Methodological details are provided at
<https://psychbruce.github.io/DPI/>. This package also includes
functions for data simulation and network analysis (correlation, partial
correlation, and Bayesian networks).

## See also

Useful links:

- <https://psychbruce.github.io/DPI/>

- Report bugs at <https://github.com/psychbruce/DPI/issues>

## Author

**Maintainer**: Han Wu Shuang Bao <baohws@foxmail.com>
([ORCID](https://orcid.org/0000-0003-3043-710X))

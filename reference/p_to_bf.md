# Convert *p* values to approximate (pseudo) Bayes Factors (PseudoBF10).

Convert *p* values to approximate (pseudo) Bayes Factors (PseudoBF10).
This transformation has been suggested by Wagenmakers (2022).

## Usage

``` r
p_to_bf(p, n, log = FALSE, label = FALSE)
```

## Arguments

- p:

  *p* value(s).

- n:

  Number of observations.

- log:

  Return `log(BF10)` or raw `BF10`. Defaults to `FALSE`.

- label:

  Add labels (i.e., names) to returned values. Defaults to `FALSE`.

## Value

A (named) numeric vector of pseudo Bayes Factors
(\\\text{PseudoBF}\_{10}\\).

## References

Wagenmakers, E.-J. (2022). *Approximate objective Bayes factors from
p-values and sample size: The \\3p\sqrt{n}\\ rule.* PsyArXiv.
[doi:10.31234/osf.io/egydq](https://doi.org/10.31234/osf.io/egydq)

## See also

[`bayestestR::p_to_bf()`](https://easystats.github.io/bayestestR/reference/p_to_bf.html)

## Examples

``` r
p_to_bf(0.05, 100)
#> [1] 0.6666667
p_to_bf(c(0.01, 0.05), 100)
#> [1] 3.3333333 0.6666667
p_to_bf(c(0.001, 0.01, 0.05, 0.1), 100, label=TRUE)
#> (p = 0.001, n = 100)  (p = 0.01, n = 100)  (p = 0.05, n = 100) 
#>           33.3333333            3.3333333            0.6666667 
#>   (p = 0.1, n = 100) 
#>            0.3333333 
p_to_bf(c(0.001, 0.01, 0.05, 0.1), 1000, label=TRUE)
#> (p = 0.001, n = 1000)  (p = 0.01, n = 1000)  (p = 0.05, n = 1000) 
#>            10.5409255             1.0540926             0.2108185 
#>   (p = 0.1, n = 1000) 
#>             0.1054093 
```

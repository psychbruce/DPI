# Simulate data from a multivariate normal distribution.

Simulate data from a multivariate normal distribution.

## Usage

``` r
sim_data(n, k, cor = NULL, exact = TRUE, seed = NULL)
```

## Arguments

- n:

  Number of observations (cases).

- k:

  Number of variables. Will be ignored if `cor` specifies a correlation
  matrix.

- cor:

  A correlation value or correlation matrix of the variables. Defaults
  to `NULL` that generates completely random data regardless of their
  empirical correlations.

- exact:

  Ensure the sample correlation matrix to be exact as specified in
  `cor`. This argument is passed on to `empirical` in
  [`mvrnorm()`](https://rdrr.io/pkg/MASS/man/mvrnorm.html). Defaults to
  `TRUE`.

- seed:

  Random seed for replicable results. Defaults to `NULL`.

## Value

Return a data.frame of simulated data.

## See also

[`cor_matrix()`](https://psychbruce.github.io/DPI/reference/cor_matrix.md)

[`sim_data_exp()`](https://psychbruce.github.io/DPI/reference/sim_data_exp.md)

## Examples

``` r
d1 = sim_data(n=100, k=5, seed=1)
cor_net(d1)
#> Displaying Correlation Network


d2 = sim_data(n=100, k=5, cor=0.2, seed=1)
cor_net(d2)
#> Displaying Correlation Network


cor.mat = cor_matrix(
  1.0, 0.7, 0.3,
  0.7, 1.0, 0.5,
  0.3, 0.5, 1.0
)
d3 = sim_data(n=100, cor=cor.mat, seed=1)
cor_net(d3)
#> Displaying Correlation Network

```

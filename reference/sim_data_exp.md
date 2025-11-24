# Simulate experiment-like data with *independent* binary Xs.

Simulate experiment-like data with *independent* binary Xs.

## Usage

``` r
sim_data_exp(
  n,
  r.xy,
  approx = TRUE,
  tol = 0.01,
  max.iter = 30,
  verbose = FALSE,
  seed = NULL
)
```

## Arguments

- n:

  Number of observations (cases).

- r.xy:

  A vector of expected correlations of each X (binary independent
  variable: 0 or 1) with Y.

- approx:

  Make the sample correlation matrix approximate more to values as
  specified in `r.xy`, using the method of orthogonal decomposition of
  residuals (i.e., making residuals more independent of Xs). Defaults to
  `TRUE`.

- tol:

  Tolerance of absolute difference between specified and empirical
  correlations. Defaults to `0.01`.

- max.iter:

  Maximum iterations for approximation. More iterations produce more
  approximate correlations, but the absolute differences will be
  convergent after about 30 iterations. Defaults to `30`.

- verbose:

  Print information about iterations that satisfy tolerance. Defaults to
  `FALSE`.

- seed:

  Random seed for replicable results. Defaults to `NULL`.

## Value

Return a data.frame of simulated data.

## See also

[`sim_data()`](https://psychbruce.github.io/DPI/reference/sim_data.md)

## Examples

``` r
data = sim_data_exp(n=1000, r.xy=c(0.5, 0.3), seed=1)
cor(data)  # tol = 0.01
#>             X1          X2         Y
#> X1 1.000000000 0.004486829 0.5053679
#> X2 0.004486829 1.000000000 0.3073449
#> Y  0.505367897 0.307344860 1.0000000

data = sim_data_exp(n=1000, r.xy=c(0.5, 0.3), seed=1,
                    verbose=TRUE)
#> 1 iterations done: abs(cor.diff.) = 0.01087 and 0.01431
#> 2 iterations done: abs(cor.diff.) = 0.00754 and 0.0101
#> 3 iterations satisfied tolerance of 0.01
cor(data)  # print iteration information
#>             X1          X2         Y
#> X1 1.000000000 0.004486829 0.5053679
#> X2 0.004486829 1.000000000 0.3073449
#> Y  0.505367897 0.307344860 1.0000000

data = sim_data_exp(n=1000, r.xy=c(0.5, 0.3), seed=1,
                    verbose=TRUE, tol=0.001)
#> 1 iterations done: abs(cor.diff.) = 0.010872 and 0.014311
#> 2 iterations done: abs(cor.diff.) = 0.007539 and 0.010095
#> 3 iterations done: abs(cor.diff.) = 0.005368 and 0.007345
#> 4 iterations done: abs(cor.diff.) = 0.003956 and 0.005555
#> 5 iterations done: abs(cor.diff.) = 0.003039 and 0.004392
#> 6 iterations done: abs(cor.diff.) = 0.002444 and 0.003637
#> 7 iterations done: abs(cor.diff.) = 0.002058 and 0.003147
#> 8 iterations done: abs(cor.diff.) = 0.001807 and 0.002829
#> 9 iterations done: abs(cor.diff.) = 0.001645 and 0.002623
#> 10 iterations done: abs(cor.diff.) = 0.00154 and 0.002489
#> 11 iterations done: abs(cor.diff.) = 0.001472 and 0.002403
#> 12 iterations done: abs(cor.diff.) = 0.001427 and 0.002347
#> 13 iterations done: abs(cor.diff.) = 0.001399 and 0.00231
#> 14 iterations done: abs(cor.diff.) = 0.00138 and 0.002287
#> 15 iterations done: abs(cor.diff.) = 0.001368 and 0.002272
#> 16 iterations done: abs(cor.diff.) = 0.00136 and 0.002262
#> 17 iterations done: abs(cor.diff.) = 0.001355 and 0.002255
#> 18 iterations done: abs(cor.diff.) = 0.001352 and 0.002251
#> 19 iterations done: abs(cor.diff.) = 0.00135 and 0.002248
#> 20 iterations done: abs(cor.diff.) = 0.001349 and 0.002247
#> 21 iterations done: abs(cor.diff.) = 0.001348 and 0.002245
#> 22 iterations done: abs(cor.diff.) = 0.001347 and 0.002245
#> 23 iterations done: abs(cor.diff.) = 0.001347 and 0.002244
#> 24 iterations done: abs(cor.diff.) = 0.001346 and 0.002244
#> 25 iterations done: abs(cor.diff.) = 0.001346 and 0.002244
#> 26 iterations done: abs(cor.diff.) = 0.001346 and 0.002244
#> 27 iterations done: abs(cor.diff.) = 0.001346 and 0.002244
#> 28 iterations done: abs(cor.diff.) = 0.001346 and 0.002244
#> 29 iterations done: abs(cor.diff.) = 0.001346 and 0.002243
#> 30 iterations done: abs(cor.diff.) = 0.001346 and 0.002243
cor(data)  # more approximate, though not exact
#>             X1          X2         Y
#> X1 1.000000000 0.004486829 0.5013461
#> X2 0.004486829 1.000000000 0.3022435
#> Y  0.501346082 0.302243457 1.0000000

data = sim_data_exp(n=1000, r.xy=c(0.5, 0.3), seed=1,
                    approx=FALSE)
cor(data)  # far less exact
#>             X1          X2         Y
#> X1 1.000000000 0.004486829 0.4793269
#> X2 0.004486829 1.000000000 0.3322344
#> Y  0.479326865 0.332234387 1.0000000
```

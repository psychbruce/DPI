# Produce a symmetric correlation matrix from values.

Produce a symmetric correlation matrix from values.

## Usage

``` r
cor_matrix(...)
```

## Arguments

- ...:

  Correlation values to transform into the symmetric correlation matrix
  (by row).

## Value

Return a symmetric correlation matrix.

## Examples

``` r
cor_matrix(
  1.0, 0.7, 0.3,
  0.7, 1.0, 0.5,
  0.3, 0.5, 1.0
)
#>      [,1] [,2] [,3]
#> [1,]  1.0  0.7  0.3
#> [2,]  0.7  1.0  0.5
#> [3,]  0.3  0.5  1.0

cor_matrix(
  1.0, NA, NA,
  0.7, 1.0, NA,
  0.3, 0.5, 1.0
)
#>      [,1] [,2] [,3]
#> [1,]  1.0   NA   NA
#> [2,]  0.7  1.0   NA
#> [3,]  0.3  0.5    1
```

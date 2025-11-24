# \[S3 methods\] for [`cor_net()`](https://psychbruce.github.io/DPI/reference/cor_net.md), [`BNs_dag()`](https://psychbruce.github.io/DPI/reference/BNs_dag.md), and [`DPI_dag()`](https://psychbruce.github.io/DPI/reference/DPI_dag.md).

- Transform `qgraph` into `ggplot`

  - `plot(cor.net)`

  - `plot(bns.dag)`

  - `plot(dpi.dag)`

- Plot network results

  - `print(cor.net)`

  - `print(bns.dag)`

  - `print(dpi.dag)`

## Usage

``` r
# S3 method for class 'cor.net'
plot(x, scale = 1.2, ...)

# S3 method for class 'cor.net'
print(x, scale = 1.2, file = NULL, width = 6, height = 4, dpi = 500, ...)

# S3 method for class 'bns.dag'
plot(x, algorithm, scale = 1.2, ...)

# S3 method for class 'bns.dag'
print(
  x,
  algorithm = names(x),
  scale = 1.2,
  file = NULL,
  width = 6,
  height = 4,
  dpi = 500,
  ...
)

# S3 method for class 'dpi.dag'
plot(
  x,
  k = min(x$DPI$k.cov),
  show.label = TRUE,
  digits.dpi = 2,
  faded.dpi = FALSE,
  faded.dpi.limit = c(0, 0.25),
  color.dpi.insig = "#EEEEEEEE",
  scale = 1.2,
  ...
)

# S3 method for class 'dpi.dag'
print(
  x,
  k = min(x$DPI$k.cov),
  show.label = TRUE,
  digits.dpi = 2,
  faded.dpi = FALSE,
  faded.dpi.limit = c(0, 0.25),
  color.dpi.insig = "#EEEEEEEE",
  scale = 1.2,
  file = NULL,
  width = 6,
  height = 4,
  dpi = 500,
  ...
)
```

## Arguments

- x:

  Object (class `cor.net` / `bns.dag` / `dpi.dag`) returned from
  [`cor_net()`](https://psychbruce.github.io/DPI/reference/cor_net.md) /
  [`BNs_dag()`](https://psychbruce.github.io/DPI/reference/BNs_dag.md) /
  [`DPI_dag()`](https://psychbruce.github.io/DPI/reference/DPI_dag.md).

- scale:

  Scale the
  [`grob`](https://wilkelab.org/cowplot/reference/draw_grob.html) object
  of `qgraph` on the `ggplot` canvas. Defaults to `1.2`.

- ...:

  Other arguments (currently not used).

- file:

  File name of saved plot (`".png"` or `".pdf"`).

- width, height:

  Width and height (in inches) of saved plot. Defaults to `6` and `4`.

- dpi:

  Dots per inch (figure resolution). Defaults to `500`.

- algorithm:

  \[For `bns.dag`\] Algorithm(s) to display. Defaults to plot the
  finally integrated DAG from BN results for each algorithm in `x`.

- k:

  \[For `dpi.dag`\] A single value of `k.cov` to produce the DPI(k) DAG.
  Defaults to `min(x$DPI$k.cov)`.

- show.label:

  \[For `dpi.dag`\] Show labels of partial correlations, DPI(k), and
  their significance on edges. Defaults to `TRUE`.

- digits.dpi:

  \[For `dpi.dag`\] Number of decimal places of DPI values displayed on
  DAG edges. Defaults to `2`.

- faded.dpi:

  \[For `dpi.dag`\] Transparency of edges according to the value of DPI.
  Defaults to `FALSE`.

- faded.dpi.limit:

  \[For `dpi.dag`\] Lower and upper limits of `abs(DPI)` for `"00"` and
  `"FF"` transparency of edges. Defaults to `c(0, 0.25)`.

- color.dpi.insig:

  \[For `dpi.dag`\] Edge color for insignificant DPIs. Defaults to
  `"#EEEEEEEE"` (faded light grey).

## Value

Return a `ggplot` object that can be further modified and used in
[`ggplot2::ggsave()`](https://ggplot2.tidyverse.org/reference/ggsave.html)
and
[`cowplot::plot_grid()`](https://wilkelab.org/cowplot/reference/plot_grid.html).

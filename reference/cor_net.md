# Correlation and partial correlation networks.

Correlation and partial correlation networks (also called Gaussian
graphical models, GGMs).

## Usage

``` r
cor_net(
  data,
  index = c("cor", "pcor"),
  show.label = TRUE,
  show.insig = FALSE,
  show.cutoff = FALSE,
  faded = FALSE,
  node.text.size = 1.2,
  node.group = NULL,
  node.color = NULL,
  edge.color.pos = "#0571B0",
  edge.color.neg = "#CA0020",
  edge.color.non = "#EEEEEEEE",
  edge.width.min = "sig",
  edge.width.max = NULL,
  edge.label.mrg = 0.01,
  file = NULL,
  width = 6,
  height = 4,
  dpi = 500,
  ...
)
```

## Arguments

- data:

  Data.

- index:

  Type of graph: `"cor"` (raw correlation network) or `"pcor"` (partial
  correlation network). Defaults to `"cor"`.

- show.label:

  Show labels of correlation coefficients and their significance on
  edges. Defaults to `TRUE`.

- show.insig:

  Show edges with insignificant correlations (*p* \> 0.05). Defaults to
  `FALSE`. To change significance level, please set `alpha` (defaults to
  `alpha=0.05`).

- show.cutoff:

  Show cut-off values of correlations. Defaults to `FALSE`.

- faded:

  Transparency of edges according to the effect size of correlation.
  Defaults to `FALSE`.

- node.text.size:

  Scalar on the font size of node (variable) labels. Defaults to `1.2`.

- node.group:

  A list that indicates which nodes belong together, with each element
  of list as a vector of integers identifying the column numbers of
  variables that belong together.

- node.color:

  A vector with a color for each element in `node.group`, or a color for
  each node.

- edge.color.pos:

  Color for (significant) positive values. Defaults to `"#0571B0"` (blue
  in ColorBrewer's RdBu palette).

- edge.color.neg:

  Color for (significant) negative values. Defaults to `"#CA0020"` (red
  in ColorBrewer's RdBu palette).

- edge.color.non:

  Color for insignificant values. Defaults to `"#EEEEEEEE"` (faded light
  grey).

- edge.width.min:

  Minimum value of edge strength to scale all edge widths. Defaults to
  `sig` (the threshold of significant values).

- edge.width.max:

  Maximum value of edge strength to scale all edge widths. Defaults to
  `NULL` (for undirected correlation networks) and `1.5` (for directed
  acyclic networks to better display arrows).

- edge.label.mrg:

  Margin of the background box around the edge label. Defaults to
  `0.01`.

- file:

  File name of saved plot (`".png"` or `".pdf"`).

- width, height:

  Width and height (in inches) of saved plot. Defaults to `6` and `4`.

- dpi:

  Dots per inch (figure resolution). Defaults to `500`.

- ...:

  Arguments passed on to
  [`qgraph()`](https://rdrr.io/pkg/qgraph/man/qgraph.html).

## Value

Return a list (class `cor.net`) of (partial) correlation results and
[`qgraph`](https://rdrr.io/pkg/qgraph/man/qgraph.html) object.

## See also

[S3method.network](https://psychbruce.github.io/DPI/reference/S3method.network.md)

[`DPI_dag()`](https://psychbruce.github.io/DPI/reference/DPI_dag.md)

[`BNs_dag()`](https://psychbruce.github.io/DPI/reference/BNs_dag.md)

## Examples

``` r
# correlation network
cor_net(airquality)
#> Displaying Correlation Network

cor_net(airquality, show.insig=TRUE)
#> Displaying Correlation Network


# partial correlation network
cor_net(airquality, "pcor")
#> Displaying Partial Correlation Network

cor_net(airquality, "pcor", show.insig=TRUE)
#> Displaying Partial Correlation Network


# modify ggplot attributes
p = cor_net(airquality, "pcor")
gg = plot(p)  # return a ggplot object
gg + labs(title="Partial Correlation Network")

```

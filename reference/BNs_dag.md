# Directed acyclic graphs (DAGs) via Bayesian networks (BNs).

Directed acyclic graphs (DAGs) via Bayesian networks (BNs). It uses
[`bnlearn::boot.strength()`](https://rdrr.io/pkg/bnlearn/man/arc.strength.html)
to estimate the strength of each edge as its *empirical frequency* over
a set of networks learned from bootstrap samples. It computes (1) the
probability of each edge (modulo its direction) and (2) the
probabilities of each edge's directions conditional on the edge being
present in the graph (in either direction). Stability thresholds are
usually set as `0.85` for *strength* (i.e., an edge appearing in more
than 85% of BNs bootstrap samples) and `0.50` for *direction* (i.e., a
direction appearing in more than 50% of BNs bootstrap samples) (Briganti
et al., 2023). Finally, for each chosen algorithm, it returns the stable
Bayesian network as the final DAG.

## Usage

``` r
BNs_dag(
  data,
  algorithm = c("pc.stable", "hc", "rsmax2"),
  algorithm.args = list(),
  n.boot = 1000,
  seed = NULL,
  strength = 0.85,
  direction = 0.5,
  node.text.size = 1.2,
  edge.width.max = 1.5,
  edge.label.mrg = 0.01,
  file = NULL,
  width = 6,
  height = 4,
  dpi = 500,
  verbose = TRUE,
  ...
)
```

## Arguments

- data:

  Data.

- algorithm:

  [Structure learning
  algorithms](https://rdrr.io/pkg/bnlearn/man/structure.learning.html)
  for building Bayesian networks (BNs). Should be function name(s) from
  the [`bnlearn`](https://rdrr.io/pkg/bnlearn/man/bnlearn-package.html)
  package. Better to perform BNs with all three classes of algorithms to
  check the robustness of results (Briganti et al., 2023).

  Defaults to the most common algorithms: `"pc.stable"` (PC), `"hc"`
  (HC), and `"rsmax2"` (RS), for the three classes, respectively.

  - \(1\) [Constraint-based
    Algorithms](https://rdrr.io/pkg/bnlearn/man/constraint.html)

    - PC:
      `"`[`pc.stable`](https://rdrr.io/pkg/bnlearn/man/constraint.html)`"`
      (*the first practical constraint-based causal structure learning
      algorithm by Peter & Clark*)

    - Others:
      `"`[`gs`](https://rdrr.io/pkg/bnlearn/man/constraint.html)`"`,
      `"`[`iamb`](https://rdrr.io/pkg/bnlearn/man/constraint.html)`"`,
      `"`[`fast.iamb`](https://rdrr.io/pkg/bnlearn/man/constraint.html)`"`,
      `"`[`inter.iamb`](https://rdrr.io/pkg/bnlearn/man/constraint.html)`"`,
      `"`[`iamb.fdr`](https://rdrr.io/pkg/bnlearn/man/constraint.html)`"`

  - \(2\) [Score-based
    Algorithms](https://rdrr.io/pkg/bnlearn/man/hc.html)

    - Hill-Climbing:
      `"`[`hc`](https://rdrr.io/pkg/bnlearn/man/hc.html)`"` (*the
      hill-climbing greedy search algorithm, exploring DAGs by
      single-edge additions, removals, and reversals, with random
      restarts to avoid local optima*)

    - Others: `"`[`tabu`](https://rdrr.io/pkg/bnlearn/man/hc.html)`"`

  - \(3\) [Hybrid
    Algorithms](https://rdrr.io/pkg/bnlearn/man/hybrid.html)
    (combination of constraint-based and score-based algorithms)

    - Restricted Maximization:
      `"`[`rsmax2`](https://rdrr.io/pkg/bnlearn/man/hybrid.html)`"`
      (*the general 2-phase restricted maximization algorithm, first
      restricting the search space and then finding the optimal
      \[maximizing the score of\] network structure in the restricted
      space*)

    - Others:
      `"`[`mmhc`](https://rdrr.io/pkg/bnlearn/man/hybrid.html)`"`,
      `"`[`h2pc`](https://rdrr.io/pkg/bnlearn/man/hybrid.html)`"`

- algorithm.args:

  An optional list of extra arguments passed to the algorithm.

- n.boot:

  Number of bootstrap samples (for learning a more "stable" network
  structure). Defaults to `1000`.

- seed:

  Random seed for replicable results. Defaults to `NULL`.

- strength:

  Stability threshold of edge *strength*: the minimum proportion
  (probability) of BNs (among the `n.boot` bootstrap samples) in which
  each edge appears.

  - Defaults to `0.85` (85%).

  - Two reverse directions share the same edge strength.

  - Empirical frequency (?~100%) will be mapped onto edge
    *width/thickness* in the final integrated `DAG`, with wider
    (thicker) edges showing stronger links, though they usually look
    similar since the default range has been limited to 0.85~1.

- direction:

  Stability threshold of edge *direction*: the minimum proportion
  (probability) of BNs (among the `n.boot` bootstrap samples) in which a
  direction of each edge appears.

  - Defaults to `0.50` (50%).

  - The proportions of two reverse directions add up to 100%.

  - Empirical frequency (?~100%) will be mapped onto edge
    *greyscale/transparency* in the final integrated `DAG`, with its
    value shown as edge text label.

- node.text.size:

  Scalar on the font size of node (variable) labels. Defaults to `1.2`.

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

- verbose:

  Print information about BN algorithm and number of bootstrap samples
  when running the analysis. Defaults to `TRUE`.

- ...:

  Arguments passed on to
  [`qgraph()`](https://rdrr.io/pkg/qgraph/man/qgraph.html).

## Value

Return a list (class `bns.dag`) of Bayesian network results and
[`qgraph`](https://rdrr.io/pkg/qgraph/man/qgraph.html) object.

## References

Briganti, G., Scutari, M., & McNally, R. J. (2023). A tutorial on
Bayesian networks for psychopathology researchers. *Psychological
Methods, 28*(4), 947–961.
[doi:10.1037/met0000479](https://doi.org/10.1037/met0000479)

Burger, J., Isvoranu, A.-M., Lunansky, G., Haslbeck, J. M. B., Epskamp,
S., Hoekstra, R. H. A., Fried, E. I., Borsboom, D., & Blanken, T. F.
(2023). Reporting standards for psychological network analyses in
cross-sectional data. *Psychological Methods, 28*(4), 806–824.
[doi:10.1037/met0000471](https://doi.org/10.1037/met0000471)

Scutari, M., & Denis, J.-B. (2021). *Bayesian networks: With examples in
R* (2nd ed.). Chapman and Hall/CRC.
[doi:10.1201/9780429347436](https://doi.org/10.1201/9780429347436)

<https://www.bnlearn.com/>

## See also

[S3method.network](https://psychbruce.github.io/DPI/reference/S3method.network.md)

[`DPI_dag()`](https://psychbruce.github.io/DPI/reference/DPI_dag.md)

[`cor_net()`](https://psychbruce.github.io/DPI/reference/cor_net.md)

## Examples

``` r
bn = BNs_dag(airquality, seed=1)
#> Warning: Missing values (NA) found in data!
#> 
#> BNs results would be affected by missing values!
#> 
#> You may use `na.omit()` to delete missing values listwise.
#> Running BN algorithm "pc.stable" with 1000 bootstrap samples...
#> Running BN algorithm "hc" with 1000 bootstrap samples...
#> Running BN algorithm "rsmax2" with 1000 bootstrap samples...
bn
#> Displaying DAG with BN algorithm "pc.stable"

#> Displaying DAG with BN algorithm "hc"

#> Displaying DAG with BN algorithm "rsmax2"

# bn$pc.stable
# bn$hc
# bn$rsmax2

## All DAG objects can be directly plotted
## or saved with print(..., file="xxx.png")
# bn$pc.stable$DAG.edge
# bn$pc.stable$DAG.strength
# bn$pc.stable$DAG.direction
# bn$pc.stable$DAG
# ...

if (FALSE) { # \dontrun{

print(bn, file="airquality.png")
# will save three plots with auto-modified file names:
- "airquality_BNs.DAG.01_pc.stable.png"
- "airquality_BNs.DAG.02_hc.png"
- "airquality_BNs.DAG.03_rsmax2.png"

# arrange multiple plots using aplot::plot_list()
# install.packages("aplot")
c1 = cor_net(airquality, "cor")
c2 = cor_net(airquality, "pcor")
bn = BNs_dag(airquality, seed=1)
mytheme = theme(plot.title=element_text(hjust=0.5))
p = aplot::plot_list(
  plot(c1),
  plot(c2),
  plot(bn$pc.stable$DAG) + mytheme,
  plot(bn$hc$DAG) + mytheme,
  plot(bn$rsmax2$DAG) + mytheme,
  design="111222
          334455",
  tag_levels="A"
)  # return a patchwork object
ggsave(p, filename="p.png", width=12, height=8, dpi=500)
ggsave(p, filename="p.pdf", width=12, height=8)
} # }
```

#### Initialize ####


#' @import ggplot2
#' @importFrom stats sd cor lm model.frame update coef
#' @importFrom stats pt pnorm rnorm quantile na.omit df.residual
#' @importFrom glue glue glue_col
#' @importFrom crayon italic underline green blue magenta
#' @importFrom grDevices png pdf dev.off
.onAttach = function(libname, pkgname) {
  inst.ver = as.character(utils::packageVersion("DPI"))
  pkg.date = substr(utils::packageDate("DPI"), 1, 4)
  packageStartupMessage(
    glue_col("

    {magenta DPI (v{inst.ver})}
    {blue The Directed Prediction Index}

    {magenta Online documentation:}
    {underline https://psychbruce.github.io/DPI}

    {magenta To use this package in publications, please cite:}
    Bao, H. W. S. ({pkg.date}). "),
    glue_col("{italic DPI: The Directed Prediction Index}"),
    glue_col(" (Version {inst.ver}) [Computer software]. "),
    glue_col("{underline https://CRAN.R-project.org/package=DPI}"),
    "\n")
}


#### Utils ####


#' Generate random data.
#'
#' @param k Number of variables.
#' @param n Number of observations (cases).
#' @param seed Random seed for replicable results.
#' Defaults to `NULL`.
#'
#' @return Return a data.frame of random data.
#'
#' @examples
#' d = data_random(k=5, n=100, seed=1)
#' cor_network(d)
#'
#' @export
data_random = function(k, n, seed=NULL) {
  set.seed(seed)
  as.data.frame(
    do.call(
      "cbind",
      lapply(
        seq_len(k),
        function(var) rnorm(n)
      )
    )
  )
}


#' Correlation network plot.
#'
#' @param data Data.
#' @param index Type of graph: `"cor"` (raw correlation network)
#' or `"pcor"` (partial correlation network).
#' Defaults to `"cor"`.
#' @param show.value Show correlation coefficients and their significance on edges.
#' Defaults to `TRUE`.
#' @param show.insig Show edges with insignificant correlations (*p* > 0.05).
#' Defaults to `FALSE`.
#' To change significance level, please set the
#' `alpha` parameter (defaults to `alpha=0.05`).
#' @param show.cutoff Show cut-off values of correlations.
#' Defaults to `FALSE`.
#' @param faded Transparency of edges according to the effect size of correlation.
#' Defaults to `FALSE`.
#' @param text.size Scalar on the font size of variable labels.
#' Defaults to `1.2`.
#' @param node.group A list that indicates which nodes belong together,
#' with each element of list as a vector of integers identifying the
#' column numbers of variables that belong together.
#' @param node.color A vector with a color for each element in `node.group`, or a color for each node.
#' @param edge.color.pos Color for (significant) positive correlations.
#' Defaults to `"#0571B0"` (blue in ColorBrewer's RdBu palette).
#' @param edge.color.neg Color for (significant) negative correlations.
#' Defaults to `"#CA0020"` (red in ColorBrewer's RdBu palette).
#' @param edge.color.insig Color for insignificant correlations.
#' Defaults to `"#EEEEEEEE"` (transparent grey).
#' @param title Plot title.
#' @param file File name of saved plot (`".png"` or `".pdf"`).
#' @param width,height Width and height (in inches) of saved plot.
#' Defaults to `6` and `4`.
#' @param dpi Dots per inch (figure resolution). Defaults to `500`.
#' @param ... Other parameters passed to \code{\link[qgraph:qgraph]{qgraph}}.
#'
#' @return Invisibly return a \code{\link[qgraph:qgraph]{qgraph}} object.
#'
#' @examples
#' cor_network(airquality)
#' cor_network(airquality, show.insig=TRUE)
#'
#' @export
cor_network = function(
    data,
    index=c("cor", "pcor"),
    show.value=TRUE,
    show.insig=FALSE,
    show.cutoff=FALSE,
    faded=FALSE,
    text.size=1.2,
    node.group=NULL,
    node.color=NULL,
    edge.color.pos="#0571B0",
    edge.color.neg="#CA0020",
    edge.color.insig="#EEEEEEEE",
    title=NULL,
    file=NULL,
    width=6,
    height=4,
    dpi=500,
    ...
) {
  index = match.arg(index)
  data = na.omit(data)
  r = cor(data)
  n = nrow(data)

  p0 = qgraph::qgraph(
    r,
    sampleSize = n,
    graph = index,
    minimum = "sig",
    DoNotPlot = TRUE
  )
  r.sig = p0[["graphAttributes"]][["Graph"]][["minimum"]]
  r.max = max(abs(p0[["Edgelist"]][["weight"]]))
  if(r.max < r.sig) show.insig = TRUE
  if(show.insig==TRUE) {
    edge.color.pos = c(edge.color.insig, edge.color.pos)
    edge.color.neg = c(edge.color.insig, edge.color.neg)
  }

  p = qgraph::qgraph(
    ## --- [data] --- ##
    r,
    sampleSize = n,
    cut = ifelse(show.insig, r.sig, 0),
    minimum = ifelse(show.insig, 0, "sig"),

    ## --- [graph] --- ##
    graph = index,
    layout = "spring",
    shape = "circle",
    # maximum = max,
    details = show.cutoff,
    title = title,

    ## --- [node] --- ##
    groups = node.group,
    color = node.color,
    palette = "ggplot2",

    ## --- [label] --- ##
    labels = names(data),
    label.cex = text.size,
    label.scale.equal = TRUE,

    ## --- [edge] --- ##
    posCol = edge.color.pos,
    negCol = edge.color.neg,
    fade = faded,
    edge.labels = show.value,
    edge.label.margin = 0.01,

    ## --- [plotting] --- ##
    usePCH = TRUE,
    DoNotPlot = TRUE,
    ...)

  if(show.value) {
    cor.values = p[["Edgelist"]][["weight"]]
    cor.labels = sprintf("%.2f", cor.values)
    cor.labels = paste0(gsub("-", "\u2013", cor.labels),
                        r_to_sig(cor.values, n))
    p[["graphAttributes"]][["Edges"]][["labels"]] = cor.labels

    edge.label.bg = p[["graphAttributes"]][["Edges"]][["label.bg"]]
    edge.color = p[["graphAttributes"]][["Edges"]][["color"]]
    edge.label.bg[edge.color==edge.color.insig] = NA
    p[["graphAttributes"]][["Edges"]][["label.bg"]] = edge.label.bg
  }

  if(!is.null(file)) {
    if(grepl("\\.png$", file))
      png(file, width=width, height=height, units="in", res=dpi)
    if(grepl("\\.pdf$", file))
      pdf(file, width=width, height=height)
  }

  plot(p)

  if(!is.null(file)) {
    dev.off()
    cli::cli_alert_success("Plot saved to {.path {paste0(getwd(), '/', file)}}")
  }

  invisible(p)
}


p.trans = function(p, digits.p=3, p.min=1e-99) {
  ifelse(
    is.na(p) | p > 1 | p < 0,
    "",
    ifelse(
      p < p.min,
      paste0("<", p.min),
      ifelse(
        p < 10^-digits.p,
        format(p, digits=1, scientific=TRUE),
        sprintf(paste0("%.", digits.p, "f"), p)
      )
    )
  )
}


sig.trans = function(p) {
  ifelse(is.na(p) | p > 1 | p < 0, "",
         ifelse(p < 0.001, "***",
                ifelse(p < 0.01, "** ",
                       ifelse(p < 0.05, "*  ",
                              ifelse(p < 0.10, ".  ", "   ")))))
}


p_to_sig = function(p) {
  ifelse(is.na(p) | p > 1 | p < 0, "",
         ifelse(p < 0.001, "***",
                ifelse(p < 0.01, "**",
                       ifelse(p < 0.05, "*", ""))))
}


r_to_sig = function(r, n) {
  p = p.t(r/sqrt((1-r^2)/(n-2)), n-2)
  p_to_sig(p)
}


p.t = function(t, df) pt(abs(t), df, lower.tail=FALSE) * 2


formula_paste = function(formula) {
  paste(formula[2], formula[1], formula[3], collapse=" ")
}


#### DPI ####


#' The Directed Prediction Index (DPI).
#'
#' The Directed Prediction Index (DPI) is
#' a simulation-based and conservative method
#' for quantifying the *relative endogeneity* (relative dependence)
#' of outcome (*Y*) vs. predictor (*X*) variables
#' in multiple linear regression models.
#' By comparing the proportion of variance explained (*R*-squared)
#' between the *Y*-as-outcome model and the *X*-as-outcome model
#' while controlling for a sufficient number
#' of potential confounding variables,
#' it suggests a more plausible influence direction
#' from a more exogenous variable (*X*)
#' to a more endogenous variable (*Y*).
#' Methodological details are provided at
#' <https://psychbruce.github.io/DPI/>.
#'
#' @param model Model object (`lm`).
#' @param y Dependent (outcome) variable.
#' @param x Independent (predictor) variable.
#' @param data \[Optional\] Defaults to `NULL`.
#' If `data` is specified, then `model` will be ignored and
#' a linear model `lm({y} ~ {x} + .)` will be fitted inside.
#' This is helpful for exploring all variables in a dataset.
#' @param k.cov Number of random covariates
#' (simulating potential omitted variables)
#' added to each simulation sample.
#'
#' - Defaults to `1`.
#' Please also test different `k.cov` values
#' as robustness checks (see [`DPI_curve`]).
#' - If `k.cov > 0`, the raw data (without bootstrapping)
#' are used, with `k.cov` random variables appended,
#' for simulation.
#' - If `k.cov = 0` (not suggested), bootstrap samples
#' (resampling with replacement) are used for simulation.
#' @param n.sim Number of simulation samples.
#' Defaults to `1000`.
#' @param seed Random seed for replicable results.
#' Defaults to `NULL`.
#' @param progress Show progress bar.
#' Defaults to `FALSE` (if `n.sim < 5000`).
#' @inheritParams cor_network
#'
#' @return Return a data.frame of simulation results:
#' - `DPI`
#'   - `t.beta.xy^2 * (R2.Y - R2.X)`
#' - `t.beta.xy`
#'   - *t* value for coefficient of X predicting Y (always equal to *t* value for coefficient of Y predicting X) when controlling for all other covariates
#' - `df.beta.xy`
#'   - residual degree of freedom (df) of `t.beta.xy`
#' - `r.partial.xy`
#'   - partial correlation (always with the same *t* value as `t.beta.xy`) between X and Y when controlling for all other covariates
#' - `delta.R2`
#'   - `R2.Y - R2.X`
#' - `R2.Y`
#'   - \eqn{R^2} of regression model predicting Y using X and all other covariates
#' - `R2.X`
#'   - \eqn{R^2} of regression model predicting X using Y and all other covariates
#'
#' @examples
#' \donttest{model = lm(Ozone ~ ., data=airquality)
#' DPI(model, y="Ozone", x="Solar.R", seed=1)
#' DPI(data=airquality, y="Ozone", x="Solar.R", k.cov=10, seed=1)
#' }
#' @export
DPI = function(
    model, y, x,
    data = NULL,
    k.cov = 1,
    n.sim = 1000,
    seed = NULL,
    progress,
    file = NULL,
    width = 6,
    height = 4,
    dpi = 500
) {
  if(missing(progress)) {
    if(n.sim < 5000)
      progress = FALSE
    else
      progress = TRUE
  }
  if(!is.null(data)) {
    model = lm(glue("{y} ~ {x} + ."), data=data)
  }
  data = model.frame(model)  # data.frame (na.omit)
  formula = formula(model)
  for(var in names(data)) {
    if(inherits(data[[var]], c("numeric", "integer", "double", "logical")) |
       (inherits(data[[var]], "factor") & nlevels(data[[var]])==2))
      data[[var]] = as.numeric(scale(as.numeric(data[[var]])))
  }
  op = options()
  options(cli.progress_bar_style="bar")
  cli::cli_progress_bar(
    clear = FALSE,
    total = n.sim,
    format = paste(
      "{cli::pb_spin} Simulation",
      "{cli::pb_current}/{cli::pb_total}",
      "{cli::pb_bar} {cli::pb_percent}",
      "[{cli::pb_elapsed_clock}]"),
    format_done = paste(
      "{cli::col_green(cli::symbol$tick)}",
      "{cli::pb_total} simulation samples estimated in {cli::pb_elapsed}")
  )
  if(!is.null(seed)) {
    set.seed(seed)
    seeds = sample(seq_len(10^8), n.sim)
  }
  set.seed(seed)
  dpi = lapply(seq_len(n.sim), function(i) {
    ## Add random covariates
    if(k.cov==0) {
      # data.i = data
      # bootstrap sample (resampling with replacement)
      data.i = data[sample(seq_len(nrow(data)), replace=TRUE),]
      covs = ""
    } else {
      if(is.null(seed))
        seed.i = NULL
      else
        seed.i = seeds[i]
      data.r = data_random(k=k.cov, n=nrow(data), seed=seed.i)
      covs = names(data.r)
      if(any(covs %in% names(data))) {
        covs = paste0("DPI_Random_Var_", covs)
        names(data.r) = covs
      }
      data.i = cbind(data, data.r)
      covs = paste("+", paste(covs, collapse=' + '))
    }
    ## Y ~ X
    model1 = update(model, formula=glue(
      "{y} ~ {x} + . {covs}"
    ), data=data.i)
    summ1 = summary(model1)
    R2.Y = summ1[["r.squared"]]
    t.xy = coef(summ1)[2, "t value"]
    rp.xy = t.xy / sqrt(t.xy^2 + df.residual(model1))  # partial r_xy
    df = df.residual(model1)
    ## X ~ Y
    model2 = update(model, formula=glue(
      "{x} ~ {y} + . {covs} - {x}"
    ), data=data.i)
    summ2 = summary(model2)
    R2.X = summ2[["r.squared"]]
    ## Return results from one random sample
    dpi = data.frame(
      DPI = t.xy^2 * (R2.Y - R2.X),
      t.beta.xy = t.xy,
      df.beta.xy = df,
      r.partial.xy = rp.xy,
      delta.R2 = R2.Y - R2.X,
      R2.Y,
      R2.X
    )
    if(progress) cli::cli_progress_update(.envir=parent.frame(2))
    return(dpi)
  })
  cli::cli_progress_done()
  options(op)
  dpi = do.call("rbind", dpi)
  class(dpi) = c("dpi", "data.frame")
  attr(dpi, "N.valid") = nrow(data)
  attr(dpi, "formula") = formula
  attr(dpi, "X") = x
  attr(dpi, "Y") = y
  attr(dpi, "k.cov") = k.cov
  attr(dpi, "n.sim") = n.sim
  attr(dpi, "df") = dpi$df.beta.xy[1]
  attr(dpi, "seed") = ifelse(is.null(seed), "NULL", seed)
  attr(dpi, "plot.params") = list(file, width, height, dpi)
  return(dpi)
}


#' \[S3 method\] Summarize DPI results.
#'
#' @param object A data.frame (of new class `dpi`)
#' returned from [`DPI`].
#' @param ... Other arguments (currently not used).
#'
#' @return Return a list with new class `summary.dpi` of
#' summarized results and raw DPI data.frame.
#'
#' @export
summary.dpi = function(object, ...) {
  ## DPI
  dpi = object$DPI
  mean = mean(dpi, na.rm=TRUE)
  se = sd(dpi, na.rm=TRUE)
  z = mean / se
  p.z = pnorm(abs(z), lower.tail=FALSE) * 2
  CIs = quantile(dpi, probs=c(0.025, 0.975), na.rm=TRUE)
  ## Delta R^2
  delta.R2 = object$delta.R2
  dR2.mean = mean(delta.R2, na.rm=TRUE)
  dR2.se = sd(delta.R2, na.rm=TRUE)
  dR2.z = dR2.mean / dR2.se
  dR2.p.z = pnorm(abs(dR2.z), lower.tail=FALSE) * 2
  dR2.CIs = quantile(delta.R2, probs=c(0.025, 0.975), na.rm=TRUE)
  ## partial r & t test (test with raw sample size!)
  r.partial = mean(object$r.partial.xy, na.rm=TRUE)
  t.r = mean(object$t.beta.xy, na.rm=TRUE)
  t.df = attr(object, "df")
  p.t = pt(abs(t.r), t.df, lower.tail=FALSE) * 2
  ## Combine
  dpi.summ = list(
    dpi.summ = data.frame(
      Estimate = mean,
      Sim.SE = se,
      z.value = z,
      p.z = p.z,
      Sim.LLCI = CIs[1],
      Sim.ULCI = CIs[2],
      row.names = "DPI"
    ),
    dR2.summ = data.frame(
      Estimate = dR2.mean,
      Sim.SE = dR2.se,
      z.value = dR2.z,
      p.z = dR2.p.z,
      Sim.LLCI = dR2.CIs[1],
      Sim.ULCI = dR2.CIs[2],
      row.names = "Delta.R2"
    ),
    r.partial.summ = data.frame(
      Estimate = r.partial,
      t.value = t.r,
      df = t.df,
      p.t = p.t,
      row.names = "r.partial"
    ),
    dpi = object
  )
  class(dpi.summ) = "summary.dpi"
  return(dpi.summ)
}


#' \[S3 method\] Print DPI summary.
#'
#' @param x A data.frame (of new class `dpi`)
#' returned from [`DPI`].
#' @param digits Number of decimal places. Defaults to `3`.
#' @param ... Other arguments (currently not used).
#'
#' @return No return value.
#'
#' @export
print.summary.dpi = function(x, digits=3, ...) {
  fmt = paste0("%.", digits, "f")
  dpi = x$dpi.summ
  res.dpi = data.frame(
    Estimate = sprintf(fmt, dpi$Estimate),
    Sim.SE = paste0("(", sprintf(fmt, dpi$Sim.SE), ")"),
    z.value = sprintf(fmt, dpi$z.value),
    p.z = p.trans(dpi$p.z, 4),
    sig = sig.trans(dpi$p.z),
    Sim.Conf.Interval = paste0(
      "[", sprintf(fmt, dpi$Sim.LLCI),
      ", ", sprintf(fmt, dpi$Sim.ULCI),
      "]"),
    row.names = "DPI"
  )
  dR2 = x$dR2.summ
  res.dR2 = data.frame(
    Estimate = sprintf(fmt, dR2$Estimate),
    Sim.SE = paste0("(", sprintf(fmt, dR2$Sim.SE), ")"),
    z.value = sprintf(fmt, dR2$z.value),
    p.z = p.trans(dR2$p.z, 4),
    sig = sig.trans(dR2$p.z),
    Sim.Conf.Interval = paste0(
      "[", sprintf(fmt, dR2$Sim.LLCI),
      ", ", sprintf(fmt, dR2$Sim.ULCI),
      "]"),
    row.names = paste0("\u0394R", cli::symbol$sup_2)
  )
  cli::cli_text("
  {cli::col_cyan('Sample size:')}
    N.valid = {attr(x$dpi, 'N.valid')}
  ")
  cli::cli_text("
  {cli::col_cyan('Model formula:')}
    {formula_paste(attr(x$dpi, 'formula'))}
  ")
  cli::cli_text("
  {cli::col_cyan('Directed prediction tested:')}
    {.val {attr(x$dpi, 'X')}} (X) -> {.val {attr(x$dpi, 'Y')}} (Y)
  ")
  cli::cli_text("
  {cli::col_cyan('Simulation sample settings:')}
    k.random.covs = {cli::col_magenta({attr(x$dpi, 'k.cov')})},
    n.sim = {cli::col_magenta({attr(x$dpi, 'n.sim')})},
    seed = {cli::col_magenta({attr(x$dpi, 'seed')})}
  ")
  cat("\n")
  cli::cli_h2("
  {cli::col_cyan('(1) Partial Correlation (pr_XY)')}
  ")
  cat(glue(
    "Estimated r(partial) = ",
    "{sprintf(fmt, x$r.partial.summ$Estimate)}, ",
    "t({x$r.partial.summ$df}) = ",
    "{sprintf(fmt, x$r.partial.summ$t.value)}, ",
    "p = {p.trans(x$r.partial.summ$p.t, 4)} ",
    "{sig.trans(x$r.partial.summ$p.t)}"
  ))
  cli::cli_h2("
  {cli::col_cyan('(2) Delta R^2 (= R^2_Y - R^2_X)')}
  ")
  print(res.dR2)
  cli::cli_h2("
  {cli::col_cyan('(3) Directed Prediction Index (DPI)')}
  ")
  print(res.dpi)
  invisible(NULL)
}


#' \[S3 method\] Plot DPI results.
#'
#' @inheritParams cor_network
#' @param x A data.frame (of new class `dpi`)
#' returned from [`DPI`].
#' @param ... Other arguments (currently not used).
#'
#' @return Return a `ggplot` object.
#'
#' @export
plot.dpi = function(x, file=NULL, width=6, height=4, dpi=500, ...) {
  DPI = scaled = ndensity = NULL
  color = "#2B579A"
  x.summ = summary(x)
  summ = x.summ$dpi.summ
  summ.r = x.summ$r.partial.summ
  r.sig = summ.r$p.t < 0.05
  if(is.null(file)) {
    plot.params = attr(x, "plot.params")
    file = plot.params$file
    width = plot.params$width
    height = plot.params$height
    dpi = plot.params$dpi
  }

  expr.x = eval(parse(text=glue("
    expression(
      paste(
        'Directed Prediction Index (',
        DPI[X %->% Y] == italic(t)[italic(r)[partial]]^2 %.% Delta,
        italic(R)[paste(Y,' vs. ',X)]^2,
        ')'
      )
    )
  ")), envir=parent.frame())

  expr.title = eval(parse(text=glue("
    expression(
      paste(
        'Histogram of DPI (',
        italic(k)[random.covs] == {attr(x, 'k.cov')},
        ', ',
        italic(n)[sim.samples] == {attr(x, 'n.sim')},
        ')'
      )
    )
  ")), envir=parent.frame())

  expr.subtitle = eval(parse(text=glue("
    expression(
      paste(
        bar(DPI)[{attr(x, 'X')} %->% {attr(x, 'Y')}],
        ' = {sprintf('%.3f', summ$Estimate)}, ',
        italic(p)[italic(z)],
        ' = {p.trans(summ$p.z)}, ',
        CI['95%']^Sim,
        ' = [{sprintf('%.3f', summ$Sim.LLCI)}',
        ', {sprintf('%.3f', summ$Sim.ULCI)}]'
      )
    )
  ")), envir=parent.frame())

  expr.caption = eval(parse(text=glue("
    expression(
      paste(
        bar(italic(r)[partial]),
        ' = {sprintf('%.3f', summ.r$Estimate)}, ',
        bar(italic(t)[italic(r)[partial]]),
        '({summ.r$df})',
        ' = {sprintf('%.3f', summ.r$t.value)}, ',
        italic(p),
        ' = {p.trans(summ.r$p.t)}'
      )
    )
  ")), envir=parent.frame())

  p = ggplot(x, aes(x=DPI)) +
    geom_histogram(
      aes(y=after_stat(ndensity)),
      bins=11, color="grey50", fill="grey", alpha=0.6) +
    geom_density(
      aes(y=after_stat(scaled)), adjust=2,  # default: adjust=1
      linewidth=0.6, color=color, fill=color, alpha=0.2) +
    geom_vline(xintercept=0, color="darkred", linetype="dashed") +
    geom_errorbarh(aes(xmin=summ$Sim.LLCI,
                       xmax=summ$Sim.ULCI,
                       y=1.03),
                   height=0.04) +
    annotate("point", x=summ$Estimate, y=1.03, shape=18, size=3,
             color=ifelse(r.sig, "black", "grey")) +
    labs(x=expr.x,
         y="Density (Scaled)",
         title=expr.title,
         subtitle=expr.subtitle,
         caption=expr.caption) +
    theme_classic() +
    theme(plot.subtitle=element_text(color=color),
          plot.caption=element_text(color="darkred"),
          axis.text=element_text(color="black"),
          axis.line=element_line(color="black"),
          axis.ticks=element_line(color="black"))

  if(!is.null(file)) {
    ggsave(p, filename=file, width=width, height=height, dpi=dpi)
    cli::cli_alert_success("Plot saved to {.path {paste0(getwd(), '/', file)}}")
  }

  return(p)
}


#' \[S3 method\] Print DPI summary and plot.
#'
#' @param x A data.frame (of new class `dpi`)
#' returned from [`DPI`].
#' @param digits Number of decimal places. Defaults to `3`.
#' @param ... Other arguments (currently not used).
#'
#' @return No return value.
#'
#' @export
print.dpi = function(x, digits=3, ...) {
  print(summary(x), digits=digits)
  print(plot(x))
}


#' The DPI curve analysis.
#'
#' @inheritParams DPI
#' @param k.covs An integer vector of number of random covariates
#' (simulating potential omitted variables)
#' added to each simulation sample.
#' Defaults to `1:10` (producing DPI results for `k.cov`=1~10).
#' For details, see [`DPI`].
#'
#' @return Return a data.frame of DPI curve results.
#'
#' @examples
#' \donttest{model = lm(Ozone ~ ., data=airquality)
#' DPIs = DPI_curve(model, y="Ozone", x="Solar.R", seed=1)
#' plot(DPIs)  # ggplot object
#' }
#' @export
DPI_curve = function(
    model, y, x,
    data = NULL,
    k.covs = 1:10,
    n.sim = 1000,
    seed = NULL,
    file = NULL,
    width = 6,
    height = 4,
    dpi = 500
) {
  op = options()
  options(cli.progress_bar_style="bar")
  cli::cli_progress_bar(
    clear = FALSE,
    total = length(k.covs),
    format = paste(
      "{cli::pb_spin} Simulation",
      "k.covs: {cli::pb_current}/{cli::pb_total}",
      "{cli::pb_bar} {cli::pb_percent}",
      "[{cli::pb_elapsed_clock}]"),
    format_done = paste(
      "{cli::col_green(cli::symbol$tick)}",
      "{cli::pb_total} * {n.sim}",
      "simulation samples estimated in {cli::pb_elapsed}")
  )
  dpi.curve = lapply(k.covs, function(k.cov) {
    dpi = DPI(model, y, x, data, k.cov, n.sim, seed, progress=FALSE)
    CIs.99 = quantile(dpi$DPI, probs=c(0.005, 0.995), na.rm=TRUE)
    dpi.summ = cbind(
      data.frame(k.cov),
      summary(dpi)[["dpi.summ"]],
      data.frame(Sim.LLCI.99 = CIs.99[1],
                 Sim.ULCI.99 = CIs.99[2])
    )
    row.names(dpi.summ) = k.cov
    cli::cli_progress_update(.envir=parent.frame(2))
    return(dpi.summ)
  })
  cli::cli_progress_done()
  options(op)
  dpi.curve = do.call("rbind", dpi.curve)
  class(dpi.curve) = c("dpi.curve", "data.frame")
  attr(dpi.curve, "X") = x
  attr(dpi.curve, "Y") = y
  attr(dpi.curve, "k.covs") = k.covs
  attr(dpi.curve, "n.sim") = n.sim
  attr(dpi.curve, "plot.params") = list(file, width, height, dpi)
  return(dpi.curve)
}


#' \[S3 method\] Plot DPI curve analysis results.
#'
#' @inheritParams cor_network
#' @param x A data.frame (of new class `dpi.curve`)
#' returned from [`DPI_curve`].
#' @param ... Other arguments (currently not used).
#'
#' @return Return a `ggplot` object.
#'
#' @export
plot.dpi.curve = function(x, file=NULL, width=6, height=4, dpi=500, ...) {
  k.cov = Estimate = Sim.LLCI = Sim.ULCI = Sim.LLCI.99 = Sim.ULCI.99 = NULL
  color = "#2B579A"
  if(is.null(file)) {
    plot.params = attr(x, "plot.params")
    file = plot.params$file
    width = plot.params$width
    height = plot.params$height
    dpi = plot.params$dpi
  }
  # dp = rbind(
  #   data.frame(
  #     k.cov = 0,
  #     Estimate = lm(Estimate ~ k.cov, x)$coefficients[1],
  #     Sim.LLCI = lm(Sim.LLCI ~ k.cov, x)$coefficients[1],
  #     Sim.ULCI = lm(Sim.ULCI ~ k.cov, x)$coefficients[1]
  #   ),
  #   x[c("k.cov", "Estimate", "Sim.LLCI", "Sim.ULCI")]
  # )
  expr.subtitle = eval(parse(text=glue("
    expression(
      paste(
        bar(DPI)[{attr(x, 'X')} %->% {attr(x, 'Y')}],
        ' with ',
        CI['95%']^Sim,
        ' (dashed) and ',
        CI['99%']^Sim,
        ' (dotted)'
      )
    )
  ")), envir=parent.frame())
  p = ggplot(x, aes(x=k.cov, y=Estimate)) +
    geom_ribbon(aes(ymin=Sim.LLCI.99, ymax=Sim.ULCI.99),
                color=color, fill=color, alpha=0.1,
                linetype="dotted") +
    geom_ribbon(aes(ymin=Sim.LLCI, ymax=Sim.ULCI),
                color=color, fill=color, alpha=0.15,
                linetype="dashed") +
    geom_path(linewidth=1, color=color) +
    geom_point(color=color, size=2) +
    geom_hline(yintercept=0, color="darkred", linetype="dashed") +
    scale_x_continuous(breaks=x$k.cov) +
    labs(
      x=paste0(
        "Number of Random Covariates (Simulation Samples = ",
        attr(x, "n.sim"),
        ")"),
      y="DPI",
      title="Directed Prediction Index (DPI) Curve Analysis",
      subtitle=expr.subtitle) +
    theme_classic() +
    theme(plot.subtitle=element_text(color=color),
          axis.text=element_text(color="black"),
          axis.line=element_line(color="black"),
          axis.ticks=element_line(color="black"))

  if(!is.null(file)) {
    ggsave(p, filename=file, width=width, height=height, dpi=dpi)
    cli::cli_alert_success("Plot saved to {.path {paste0(getwd(), '/', file)}}")
  }

  return(p)
}


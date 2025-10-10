#### Initialize ####


#' @keywords internal
"_PACKAGE"


#' @import ggplot2
#' @importFrom stats var sd cor na.omit
#' @importFrom stats pt pnorm rnorm rbinom quantile qnorm
#' @importFrom stats lm model.frame update coef df.residual
#' @importFrom glue glue glue_col
#' @importFrom crayon italic underline green blue magenta
#' @importFrom cowplot draw_grob as_grob
.onAttach = function(libname, pkgname) {
  inst.ver = as.character(utils::packageVersion("DPI"))
  pkgs = c("ggplot2", "cowplot")
  suppressMessages({
    suppressWarnings({
      loaded = sapply(pkgs, require, character.only=TRUE)
    })
  })
  packageStartupMessage(
    glue::glue_col("

    {magenta DPI (v{inst.ver})}
    {blue The Directed Prediction Index}

    {magenta Packages also loaded:}
    {green \u2714 ggplot2, cowplot}

    {magenta Online documentation:}
    {underline https://psychbruce.github.io/DPI}

    {magenta To use this package in publications, please cite:}
    Bao, H. W. S. (2025). {italic DPI: The Directed Prediction Index for causal inference from observational data} (Version {inst.ver}) [Computer software]. {underline https://doi.org/10.32614/CRAN.package.DPI}

    "))
}


#### Utils ####


as_numeric = function(data) {
  data = as.data.frame(data)
  for(var in names(data)) {
    if(inherits(data[[var]], c("numeric", "integer", "double", "logical")) |
       (inherits(data[[var]], "factor") & nlevels(data[[var]])==2))
      data[[var]] = as.numeric(data[[var]])
  }
  return(data)
}


#' Simulate data from a multivariate normal distribution.
#'
#' @param n Number of observations (cases).
#' @param k Number of variables. Will be ignored if `cor` specifies a correlation matrix.
#' @param cor A correlation value or correlation matrix of the variables. Defaults to `NULL` that generates completely random data regardless of their empirical correlations.
#' @param exact Ensure the sample correlation matrix to be exact as specified in `cor`. This argument is passed on to `empirical` in [`mvrnorm()`][MASS::mvrnorm]. Defaults to `TRUE`.
#' @param seed Random seed for replicable results. Defaults to `NULL`.
#'
#' @return
#' Return a data.frame of simulated data.
#'
#' @seealso
#' [cor_matrix()]
#'
#' [sim_data_exp()]
#'
#' @examples
#' d1 = sim_data(n=100, k=5, seed=1)
#' cor_net(d1)
#'
#' d2 = sim_data(n=100, k=5, cor=0.2, seed=1)
#' cor_net(d2)
#'
#' cor.mat = cor_matrix(
#'   1.0, 0.7, 0.3,
#'   0.7, 1.0, 0.5,
#'   0.3, 0.5, 1.0
#' )
#' d3 = sim_data(n=100, cor=cor.mat, seed=1)
#' cor_net(d3)
#'
#' @export
sim_data = function(n, k, cor=NULL, exact=TRUE, seed=NULL) {
  if(is.null(cor)) {
    set.seed(seed)
    data = as.data.frame(
      do.call(
        "cbind",
        lapply(
          seq_len(k),
          function(var) rnorm(n)
        )
      )
    )
  } else {
    if(is.matrix(cor)) {
      k = ncol(cor)
    } else {
      if(length(cor)!=1)
        stop("`cor` must be a single correlation value or a correlation matrix.", call.=FALSE)
      cor = matrix(rep(cor, k^2), nrow=k, byrow=TRUE)
      diag(cor) = 1
    }
    set.seed(seed)
    data = as.data.frame(
      MASS::mvrnorm(
        n = n,
        mu = rep(0, k),
        Sigma = cor,
        empirical = exact
      )
    )
  }
  return(data)
}


#' Produce a symmetric correlation matrix from values.
#'
#' @param ... Correlation values to transform into the symmetric correlation matrix (by row).
#'
#' @return
#' Return a symmetric correlation matrix.
#'
#' @examples
#' cor_matrix(
#'   1.0, 0.7, 0.3,
#'   0.7, 1.0, 0.5,
#'   0.3, 0.5, 1.0
#' )
#'
#' cor_matrix(
#'   1.0, NA, NA,
#'   0.7, 1.0, NA,
#'   0.3, 0.5, 1.0
#' )
#'
#' @export
cor_matrix = function(...) {
  cor.vec = as.numeric(unlist(list(...)))
  matrix(cor.vec, nrow=sqrt(length(cor.vec)), byrow=TRUE)
}


#' Simulate experiment-like data with *independent* binary Xs.
#'
#' @inheritParams sim_data
#' @param r.xy A vector of expected correlations of each X (binary independent variable: 0 or 1) with Y.
#' @param approx Make the sample correlation matrix approximate more to values as specified in `r.xy`, using the method of orthogonal decomposition of residuals (i.e., making residuals more independent of Xs). Defaults to `TRUE`.
#' @param tol Tolerance of absolute difference between specified and empirical correlations. Defaults to `0.01`.
#' @param max.iter Maximum iterations for approximation. More iterations produce more approximate correlations, but the absolute differences will be convergent after about 30 iterations. Defaults to `30`.
#' @param verbose Print information about iterations that satisfy tolerance. Defaults to `FALSE`.
#'
#' @return
#' Return a data.frame of simulated data.
#'
#' @seealso
#' [sim_data()]
#'
#' @examples
#' \donttest{data = sim_data_exp(n=1000, r.xy=c(0.5, 0.3), seed=1)
#' cor(data)  # tol = 0.01
#'
#' data = sim_data_exp(n=1000, r.xy=c(0.5, 0.3), seed=1,
#'                     verbose=TRUE)
#' cor(data)  # print iteration information
#'
#' data = sim_data_exp(n=1000, r.xy=c(0.5, 0.3), seed=1,
#'                     verbose=TRUE, tol=0.001)
#' cor(data)  # more approximate, though not exact
#'
#' data = sim_data_exp(n=1000, r.xy=c(0.5, 0.3), seed=1,
#'                     approx=FALSE)
#' cor(data)  # far less exact
#' }
#' @export
sim_data_exp = function(n, r.xy, approx=TRUE, tol=0.01, max.iter=30, verbose=FALSE, seed=NULL) {
  ids = seq_along(r.xy)  # 1, 2, 3, ...

  set.seed(seed)
  Xs = do.call(cbind, lapply(ids, function(i) {
    # completely independent binary Xs
    rbinom(n, size=1, prob=0.5)  # random (50%): 0 or 1
  }))
  colnames(Xs) = paste0("X", ids)
  X = cbind(Intercept=1, Xs)

  B = sapply(ids, function(i) {
    # var.y = 1
    # r.xy = B * SD.x / SD.y = (M1 - M0) * sqrt(p*(1-p)) / SD.y
    # p = mean(Xs[,i])
    # SD.x = sqrt(p*(1-p))
    SD.x = sd(Xs[,i])
    SD.y = 1
    B = r.xy[i] * SD.y / SD.x
    names(B) = paste0("b", i)
    return(B)
  })
  B = c(b0=0, B)

  Y.pred = X %*% B
  # var.y = 1 = var.pred + var.residual
  sd.residual = drop(sqrt(1 - var(Y.pred)))
  if(is.na(sd.residual))
    stop("Invalid specification producing NaN in outcome. Perhaps r.xy is too large?", call.=FALSE)
  res = rnorm(n, mean=0, sd=sd.residual)
  if(approx) {
    # orthogonal decomposition:
    # make residuals more independent of Xs
    proj = Xs %*% solve(t(Xs) %*% Xs) %*% t(Xs)
    for(iter in seq_len(max.iter)) {
      res = res - proj %*% res
      res = scale(res) * sd.residual
      cor = cor(cbind(Xs, Y.pred + res))
      diff.cor.abs = abs(cor[nrow(cor), -ncol(cor)] - r.xy)
      if(all(diff.cor.abs < tol)) {
        if(verbose)
          cli::cli_text("{.val {iter}} iterations satisfied tolerance of {.val {tol}}")
        break
      }
      if(verbose)
        cli::cli_text("{.val {iter}} iterations done: abs(cor.diff.) = {.val {round(diff.cor.abs, 3-log10(tol))}}")
    }
  }
  Y = Y.pred + res
  data = data.frame(Xs, Y)
  return(data)
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


p.t = function(t, df) {
  pt(abs(t), df, lower.tail=FALSE) * 2
}


r_to_p = function(r, df) {
  p.t(r/sqrt((1-r^2)/df), df)
}


formula_paste = function(formula) {
  paste(formula[2], formula[1], formula[3], collapse=" ")
}


file_insert_name = function(file, name) {
  file = strsplit(file, "/")[[1]]
  file[length(file)] = paste0(
    file_ext(file[length(file)], "txt"),
    name,
    file_ext(file[length(file)], "ext")
  )
  file = paste(file, collapse="/")
  return(file)
}


file_ext = function(file, return=c("ext", "txt")) {
  return = match.arg(return)
  pos = regexpr("\\.([[:alnum:]]+)$", file)
  if(return=="ext")
    return(ifelse(pos > -1L, tolower(substring(file, pos)), ""))
  if(return=="txt")
    return(substring(file, 1L, last=pos-1L))
}


#### DPI ####


#' \[S3 methods\] for [DPI()] and [DPI_curve()].
#'
#' \describe{
#'   \item{`summary(dpi)`}{
#'     Summarize DPI results.
#'     Return a list (class `summary.dpi`) of summarized results and raw DPI data.frame.
#'   }
#'   \item{`print(summary.dpi)`}{
#'     Print DPI summary.
#'   }
#'   \item{`plot(dpi)`}{
#'     Plot DPI results.
#'     Return a `ggplot` object.
#'   }
#'   \item{`print(dpi)`}{
#'     Print DPI summary and plot.
#'   }
#'   \item{`plot(dpi.curve)`}{
#'     Plot DPI curve analysis results.
#'     Return a `ggplot` object.
#'   }
#' }
#' @keywords internal
#' @name S3method.dpi
#' @inheritParams DPI
#' @param object Object (class `dpi`) returned from [DPI()].
#' @param x Object (class `dpi` or `dpi.curve`) returned from [DPI()] or [DPI_curve()].
#' @param digits Number of decimal places. Defaults to `3`.
#' @param ... Other arguments (currently not used).
NULL


#' The Directed Prediction Index (DPI).
#'
#' The Directed Prediction Index (DPI) is a quasi-causal inference method for cross-sectional data designed to quantify the *relative endogeneity* (relative dependence) of outcome (*Y*) vs. predictor (*X*) variables in regression models. By comparing the proportion of variance explained (*R*-squared) between the *Y*-as-outcome model and the *X*-as-outcome model while controlling for a sufficient number of possible confounders, it can suggest a plausible (admissible) direction of influence from a more exogenous variable (*X*) to a more endogenous variable (*Y*). Methodological details are provided at <https://psychbruce.github.io/DPI/>.
#'
#' @param model Model object (`lm`).
#' @param y Dependent (outcome) variable.
#' @param x Independent (predictor) variable.
#' @param data \[Optional\] Defaults to `NULL`. If `data` is specified, then `model` will be ignored and a linear model `lm({y} ~ {x} + .)` will be fitted inside. This is helpful for exploring all variables in a dataset.
#' @param k.cov Number of random covariates (simulating potential omitted variables) added to each simulation sample.
#'
#' - Defaults to `1`. Please also test different `k.cov` values as robustness checks (see [DPI_curve()]).
#' - If `k.cov` > 0, the raw data (without bootstrapping) are used, with `k.cov` random variables appended, for simulation.
#' - If `k.cov` = 0 (not suggested), bootstrap samples (resampling with replacement) are used for simulation.
#' @param n.sim Number of simulation samples. Defaults to `1000`.
#' @param alpha Significance level for computing the `Strength` score (0~1) based on *p* value of partial correlation between `X` and `Y`. Defaults to `0.05`.
#' - `Direction = R2.Y - R2.X`
#' - `Strength = 1 - tanh(p.beta.xy/alpha/2)`
#' @param bonf Bonferroni correction to control for false positive rates: `alpha` is divided by, and *p* values are multiplied by, the number of comparisons.
#' - Defaults to `FALSE`: No correction, suitable if you plan to test only one pair of variables.
#' - `TRUE`: Using `k * (k - 1) / 2` (number of all combinations of variable pairs) where `k = length(data)`.
#' - A user-specified number of comparisons.
#' @param seed Random seed for replicable results. Defaults to `NULL`.
#' @param progress Show progress bar. Defaults to `FALSE` (if `n.sim` < 5000).
#' @param file File name of saved plot (`".png"` or `".pdf"`).
#' @param width,height Width and height (in inches) of saved plot. Defaults to `6` and `4`.
#' @param dpi Dots per inch (figure resolution). Defaults to `500`.
#'
#' @return
#' Return a data.frame of simulation results:
#' - `DPI`
#'   - `= Direction * Strength`
#'   - `= (R2.Y - R2.X) * (1 - tanh(p.beta.xy/alpha/2))`
#' - `delta.R2`
#'   - `R2.Y - R2.X`
#' - `R2.Y`
#'   - \eqn{R^2} of regression model predicting Y using X and all other covariates
#' - `R2.X`
#'   - \eqn{R^2} of regression model predicting X using Y and all other covariates
#' - `t.beta.xy`
#'   - *t* value for coefficient of X predicting Y (always equal to *t* value for coefficient of Y predicting X) when controlling for all other covariates
#' - `p.beta.xy`
#'   - *p* value for coefficient of X predicting Y (always equal to *p* value for coefficient of Y predicting X) when controlling for all other covariates
#' - `df.beta.xy`
#'   - residual degree of freedom (df) of `t.beta.xy`
#' - `r.partial.xy`
#'   - partial correlation (always with the same *t* value as `t.beta.xy`) between X and Y when controlling for all other covariates
#'
#' @seealso
#' [S3method.dpi]
#'
#' [DPI_curve()]
#'
#' [DPI_dag()]
#'
#' [BNs_dag()]
#'
#' [cor_net()]
#'
#' @examples
#' \donttest{# input a fitted model
#' model = lm(Ozone ~ ., data=airquality)
#' DPI(model, y="Ozone", x="Solar.R", seed=1)  # DPI > 0
#' DPI(model, y="Ozone", x="Wind", seed=1)     # DPI > 0
#' DPI(model, y="Wind", x="Solar.R", seed=1)   # unrelated
#'
#' # input raw data, test with more random covs
#' DPI(data=airquality, y="Ozone", x="Solar.R", k.cov=10, seed=1)
#' DPI(data=airquality, y="Ozone", x="Wind", k.cov=10, seed=1)
#' DPI(data=airquality, y="Wind", x="Solar.R", k.cov=10, seed=1)
#' }
#' @export
DPI = function(
    model, y, x,
    data = NULL,
    k.cov = 1,
    n.sim = 1000,
    alpha = 0.05,
    bonf = FALSE,
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

  formula.y = glue("{y} ~ {x} + .")
  formula.x = glue("{x} ~ {y} + .")

  if(is.null(data)) {
    data = model.frame(model)  # new data.frame (na.omit)
    model = lm(formula.y, data=data)  # refit
  } else {
    data = as_numeric(data)
    model = lm(formula.y, data=data)
    data = model.frame(model)  # new data.frame (na.omit)
  }

  formula = formula(model)
  formula.y = update(formula, glue("{y} ~ {x} + . - {y}"))
  formula.x = update(formula, glue("{x} ~ {y} + . - {x}"))

  if(is.numeric(bonf)) {
    alpha = alpha / bonf
  } else {
    if(bonf) {
      k = length(data)
      bonf = k * (k - 1) / 2
      alpha = alpha / bonf
    } else {
      bonf = 1
    }
  }
  bonf = as.integer(bonf)

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
      data.r = sim_data(n=nrow(data), k=k.cov, seed=seed.i)
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
      "{y} ~ {x} + . {covs} - {y}"
    ), data=data.i)
    summ1 = summary(model1)
    R2.Y = summ1[["r.squared"]]
    t.xy = coef(summ1)[2, "t value"]
    p.xy = coef(summ1)[2, "Pr(>|t|)"]
    df = df.residual(model1)
    rp.xy = t.xy / sqrt(t.xy^2 + df)  # partial r_xy
    ## X ~ Y
    model2 = update(model, formula=glue(
      "{x} ~ {y} + . {covs} - {x}"
    ), data=data.i)
    summ2 = summary(model2)
    R2.X = summ2[["r.squared"]]
    ## Return results from one random sample
    dpi = data.frame(
      # DPI = t.xy^2 * (R2.Y - R2.X),
      DPI = (R2.Y - R2.X) * (1 - tanh(p.xy/alpha/2)),
      # using plogis: 2 * (1 - plogis(p, scale=alpha))
      # using sigmoid: 2 * ( 1 - 1 / (1 + exp(-p/alpha)) )
      # plogis(): logistic distribution function, "inverse logit"
      #   - plogis() is also a rescaled hyperbolic tangent:
      #       plogis(x) = sigmoid(x) = 1 / (1 + exp(-x))
      #         = (1 + tanh(x/2)) / 2
      #       plogis(x, scale) = (1 + tanh(x/scale/2)) / 2
      # p = seq(0, 1, 0.01)
      # plot(p, 1 - tanh(p/0.05))
      delta.R2 = R2.Y - R2.X,
      R2.Y,
      R2.X,
      t.beta.xy = t.xy,
      p.beta.xy = p.xy,
      df.beta.xy = df,
      r.partial.xy = rp.xy
    )
    if(progress)
      cli::cli_progress_update(.envir=parent.frame(2))
    return(dpi)
  })
  cli::cli_progress_done()
  options(op)
  gc()
  dpi = do.call("rbind", dpi)
  class(dpi) = c("dpi", "data.frame")
  attr(dpi, "N.valid") = nrow(data)
  attr(dpi, "df") = dpi$df.beta.xy[1]
  attr(dpi, "formula.y") = formula.y
  attr(dpi, "formula.x") = formula.x
  attr(dpi, "X") = x
  attr(dpi, "Y") = y
  attr(dpi, "k.cov") = k.cov
  attr(dpi, "n.sim") = n.sim
  attr(dpi, "alpha") = alpha
  attr(dpi, "bonferroni") = bonf
  attr(dpi, "seed") = ifelse(is.null(seed), "NULL", seed)
  attr(dpi, "plot.params") = list(file = file,
                                  width = width,
                                  height = height,
                                  dpi = dpi)
  return(dpi)
}


#' @rdname S3method.dpi
#' @export
summary.dpi = function(object, ...) {
  ## DPI
  alpha = attr(object, "alpha")
  bonf = attr(object, "bonferroni")
  dpi = object$DPI
  mean = mean(dpi, na.rm=TRUE)
  se = sd(dpi, na.rm=TRUE)
  z = mean / se
  p.z = min(1, pnorm(abs(z), lower.tail=FALSE) * 2 * bonf)
  # CIs = quantile(dpi, probs=c(0.025, 0.975), na.rm=TRUE)
  CIs = mean + qnorm(c(alpha/2, 1-alpha/2)) * se

  ## Delta R^2
  delta.R2 = object$delta.R2
  dR2.mean = mean(delta.R2, na.rm=TRUE)
  dR2.se = sd(delta.R2, na.rm=TRUE)
  dR2.z = dR2.mean / dR2.se
  dR2.p.z = min(1, pnorm(abs(dR2.z), lower.tail=FALSE) * 2 * bonf)
  # dR2.CIs = quantile(delta.R2, probs=c(0.025, 0.975), na.rm=TRUE)
  dR2.CIs = dR2.mean + qnorm(c(alpha/2, 1-alpha/2)) * dR2.se

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


#' @rdname S3method.dpi
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
    Conf.Interval = paste0(
      "[", sprintf(fmt, dpi$Sim.LLCI),
      ", ", sprintf(fmt, dpi$Sim.ULCI),
      "]"),
    row.names = "DPI"
  )

  cli::cli_text(
    cli::col_cyan("Sample size: "),
    "N.valid = {attr(x$dpi, 'N.valid')}")
  cli::cli_text(
    cli::col_cyan("Model Y formula: "),
    "{formula_paste(attr(x$dpi, 'formula.y'))}")
  cli::cli_text(
    cli::col_cyan("Model X formula: "),
    "{formula_paste(attr(x$dpi, 'formula.x'))}")
  cli::cli_text(
    cli::col_cyan("Directed prediction: "),
    "{.val {attr(x$dpi, 'X')}} (X) -> {.val {attr(x$dpi, 'Y')}} (Y)")
  cli::cli_text(
    cli::col_cyan("Partial correlation: "),
    "r(partial).XY = {cli::col_yellow({sprintf(fmt, x$r.partial.summ$Estimate)})},
     p = {cli::col_yellow({p.trans(x$r.partial.summ$p.t, 4)})}
     {cli::col_yellow({sig.trans(x$r.partial.summ$p.t)})}")
  cli::cli_text(
    cli::col_cyan("Simulation sample settings: "),
    "k.random.covs = {cli::col_magenta({attr(x$dpi, 'k.cov')})},
     n.sim = {cli::col_magenta({attr(x$dpi, 'n.sim')})},
     seed = {cli::col_magenta({attr(x$dpi, 'seed')})}")
  cli::cli_text(
    cli::col_cyan("False positive rates (FPR) control: "),
    "Alpha = {cli::col_magenta({format(attr(x$dpi, 'alpha'), digits=digits)})}
     (Bonferroni correction = {cli::col_magenta({attr(x$dpi, 'bonferroni')})})")
  print(res.dpi)
  invisible(NULL)
}


#' @rdname S3method.dpi
#' @export
plot.dpi = function(x, file=NULL, width=6, height=4, dpi=500, ...) {
  DPI = scaled = ndensity = NULL
  color = "#2B579A"
  x.summ = summary(x)
  summ = x.summ$dpi.summ
  summ.r = x.summ$r.partial.summ
  r.sig = summ.r$p.t < 0.05
  bonf = attr(x, "bonferroni")
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
        'Directed Prediction Index: ',
        DPI[X %->% Y] == (italic(R)[italic(Y)]^2 - italic(R)[italic(X)]^2) %.% (1 - plain(tanh) * frac(italic(p)[paste(italic(XY), '|', italic(Covs))], 2 * alpha))
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
        {ifelse(bonf==1, \"italic(p)[italic(z)]\",
                paste0(\"italic(p)[italic(z)]^'Bonf=\", bonf, \"'\"))},
        ' = {p.trans(summ$p.z)}, ',
        {ifelse(bonf==1, \"CI['95%']\",
                paste0(\"CI['95%']^'Bonf=\", bonf, \"'\"))},
        ' = [{sprintf('%.3f', summ$Sim.LLCI)}',
        ', {sprintf('%.3f', summ$Sim.ULCI)}]'
      )
    )
  ")), envir=parent.frame())

  expr.caption = eval(parse(text=glue("
    expression(
      paste(
        bar(italic(r)[partial[italic(XY)]]),
        ' = {sprintf('%.3f', summ.r$Estimate)}, ',
        bar(italic(t)[italic(r)[partial]]),
        '({summ.r$df})',
        ' = {sprintf('%.3f', summ.r$t.value)}, ',
        italic(p)[italic(r)[partial]],
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
                   width=0.04) +
    # ggplot2 update: `height` was translated to `width`.
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
    cli::cli_alert_success("Plot saved to {.path {file}}")
  }

  return(p)
}


#' @rdname S3method.dpi
#' @export
print.dpi = function(x, digits=3, ...) {
  print(summary(x), digits=digits)
  print(plot(x))
}


#' DPI curve analysis across multiple random covariates.
#'
#' @inheritParams DPI
#' @param k.covs An integer vector of number of random covariates (simulating potential omitted variables) added to each simulation sample. Defaults to `1:10` (producing DPI results for `k.cov`=1~10). For details, see [DPI()].
#' @param progress Show progress bar. Defaults to `TRUE` (if `length(k.covs)` >= 5).
#'
#' @return
#' Return a data.frame of DPI curve results.
#'
#' @seealso
#' [S3method.dpi]
#'
#' [DPI()]
#'
#' [DPI_dag()]
#'
#' [BNs_dag()]
#'
#' [cor_net()]
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
    alpha = 0.05,
    bonf = FALSE,
    seed = NULL,
    progress,
    file = NULL,
    width = 6,
    height = 4,
    dpi = 500
) {
  if(missing(progress)) {
    if(length(k.covs) < 5)
      progress = FALSE
    else
      progress = TRUE
  }
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
    dpi = DPI(model, y, x, data,
              k.cov, n.sim,
              alpha, bonf,
              seed, progress=FALSE)
    dpi.summ = summary(dpi)[["dpi.summ"]]
    # CIs.99 = quantile(dpi$DPI, probs=c(0.005, 0.995), na.rm=TRUE)
    # CIs.99 = dpi.summ$Estimate + qnorm(c(0.005, 0.995)) * dpi.summ$Sim.SE
    dpi.summ = cbind(
      data.frame(k.cov),
      dpi.summ
      # data.frame(Sim.LLCI.99 = CIs.99[1],
      #            Sim.ULCI.99 = CIs.99[2])
    )
    row.names(dpi.summ) = k.cov
    attr(dpi.summ, "bonferroni") = attr(dpi, "bonferroni")
    if(progress)
      cli::cli_progress_update(.envir=parent.frame(2))
    return(dpi.summ)
  })
  cli::cli_progress_done()
  options(op)
  gc()
  dpi.curve = do.call("rbind", dpi.curve)
  class(dpi.curve) = c("dpi.curve", "data.frame")
  attr(dpi.curve, "X") = x
  attr(dpi.curve, "Y") = y
  attr(dpi.curve, "k.covs") = k.covs
  attr(dpi.curve, "n.sim") = n.sim
  attr(dpi.curve, "plot.params") = list(file = file,
                                        width = width,
                                        height = height,
                                        dpi = dpi)
  return(dpi.curve)
}


#' @rdname S3method.dpi
#' @export
plot.dpi.curve = function(x, file=NULL, width=6, height=4, dpi=500, ...) {
  k.cov = Estimate = Sim.LLCI = Sim.ULCI = Sim.LLCI.99 = Sim.ULCI.99 = NULL
  color = "#2B579A"

  bonf = attr(x, "bonferroni")

  if(is.null(file)) {
    plot.params = attr(x, "plot.params")
    file = plot.params$file
    width = plot.params$width
    height = plot.params$height
    dpi = plot.params$dpi
  }

  expr.subtitle = eval(parse(text=glue("
    expression(
      paste(
        bar(DPI)[{attr(x, 'X')} %->% {attr(x, 'Y')}],
        ' with ',
        {ifelse(bonf==1, \"CI['95%']\",
                paste0(\"CI['95%']^'Bonf=\", bonf, \"'\"))}
      )
    )
  ")), envir=parent.frame())

  p = ggplot(x, aes(x=k.cov, y=Estimate)) +
    # geom_ribbon(aes(ymin=Sim.LLCI.99, ymax=Sim.ULCI.99),
    #             color=color, fill=color, alpha=0.1,
    #             linetype="dotted") +
    geom_ribbon(aes(ymin=Sim.LLCI, ymax=Sim.ULCI),
                color=color, fill=color, alpha=0.2,
                linetype="dashed") +
    geom_line(linewidth=1, color=color) +
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
    cli::cli_alert_success("Plot saved to {.path {file}}")
  }

  return(p)
}


#### Network ####


#' \[S3 methods\] for [cor_net()], [BNs_dag()], and [DPI_dag()].
#'
#' - Transform `qgraph` into `ggplot`
#'   - `plot(cor.net)`
#'   - `plot(bns.dag)`
#'   - `plot(dpi.dag)`
#' - Plot network results
#'   - `print(cor.net)`
#'   - `print(bns.dag)`
#'   - `print(dpi.dag)`
#' @keywords internal
#' @name S3method.network
#' @inheritParams cor_net
#' @inheritParams BNs_dag
#' @param x Object (class `cor.net` / `bns.dag` / `dpi.dag`) returned from [cor_net()] / [BNs_dag()] / [DPI_dag()].
#' @param scale Scale the [`grob`][cowplot::draw_grob] object of `qgraph` on the `ggplot` canvas. Defaults to `1.2`.
#' @param ... Other arguments (currently not used).
#' @return
#' Return a `ggplot` object that can be further modified and used in [ggplot2::ggsave()] and [cowplot::plot_grid()].
NULL


#' Correlation and partial correlation networks.
#'
#' Correlation and partial correlation networks (also called Gaussian graphical models, GGMs).
#'
#' @param data Data.
#' @param index Type of graph: `"cor"` (raw correlation network) or `"pcor"` (partial correlation network). Defaults to `"cor"`.
#' @param show.label Show labels of correlation coefficients and their significance on edges. Defaults to `TRUE`.
#' @param show.insig Show edges with insignificant correlations (*p* > 0.05). Defaults to `FALSE`. To change significance level, please set `alpha` (defaults to `alpha=0.05`).
#' @param show.cutoff Show cut-off values of correlations. Defaults to `FALSE`.
#' @param faded Transparency of edges according to the effect size of correlation. Defaults to `FALSE`.
#' @param node.text.size Scalar on the font size of node (variable) labels. Defaults to `1.2`.
#' @param node.group A list that indicates which nodes belong together, with each element of list as a vector of integers identifying the column numbers of variables that belong together.
#' @param node.color A vector with a color for each element in `node.group`, or a color for each node.
#' @param edge.color.pos Color for (significant) positive values. Defaults to `"#0571B0"` (blue in ColorBrewer's RdBu palette).
#' @param edge.color.neg Color for (significant) negative values. Defaults to `"#CA0020"` (red in ColorBrewer's RdBu palette).
#' @param edge.color.non Color for insignificant values. Defaults to `"#EEEEEEEE"` (faded light grey).
#' @param edge.width.min Minimum value of edge strength to scale all edge widths. Defaults to `sig` (the threshold of significant values).
#' @param edge.width.max Maximum value of edge strength to scale all edge widths. Defaults to `NULL` (for undirected correlation networks) and `1.5` (for directed acyclic networks to better display arrows).
#' @param edge.label.mrg Margin of the background box around the edge label. Defaults to `0.01`.
#' @param file File name of saved plot (`".png"` or `".pdf"`).
#' @param width,height Width and height (in inches) of saved plot. Defaults to `6` and `4`.
#' @param dpi Dots per inch (figure resolution). Defaults to `500`.
#' @param ... Arguments passed on to [`qgraph()`][qgraph::qgraph].
#'
#' @return
#' Return a list (class `cor.net`) of (partial) correlation results and [`qgraph`][qgraph::qgraph] object.
#'
#' @seealso
#' [S3method.network]
#'
#' [DPI_dag()]
#'
#' [BNs_dag()]
#'
#' @examples
#' \donttest{# correlation network
#' cor_net(airquality)
#' cor_net(airquality, show.insig=TRUE)
#'
#' # partial correlation network
#' cor_net(airquality, "pcor")
#' cor_net(airquality, "pcor", show.insig=TRUE)
#'
#' # modify ggplot attributes
#' p = cor_net(airquality, "pcor")
#' gg = plot(p)  # return a ggplot object
#' gg + labs(title="Partial Correlation Network")
#' }
#' @export
cor_net = function(
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
) {
  index = match.arg(index)
  data = as_numeric(na.omit(data))
  r = cor(data)
  n = nrow(data)
  k = ncol(data)
  df = n - k

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
    edge.color.pos = c(edge.color.non, edge.color.pos)
    edge.color.neg = c(edge.color.non, edge.color.neg)
  }

  p = qgraph::qgraph(
    ## --- [data] --- ##
    r,
    sampleSize = n,
    cut = ifelse(show.insig, r.sig, 0),
    minimum = ifelse(show.insig, 0, edge.width.min),
    maximum = edge.width.max,

    ## --- [graph] --- ##
    graph = index,
    layout = "spring",
    shape = "circle",
    # maximum = max,
    details = show.cutoff,

    ## --- [node] --- ##
    groups = node.group,
    color = node.color,
    palette = "ggplot2",

    ## --- [label] --- ##
    labels = names(data),
    label.cex = node.text.size,
    label.scale.equal = TRUE,

    ## --- [edge] --- ##
    posCol = edge.color.pos,
    negCol = edge.color.neg,
    fade = faded,
    edge.labels = show.label,
    edge.label.margin = edge.label.mrg,

    ## --- [plotting] --- ##
    usePCH = TRUE,
    DoNotPlot = TRUE,
    ...)

  vars = p[["Arguments"]][["labels"]]
  vars.from = p[["Edgelist"]][["from"]]
  vars.to = p[["Edgelist"]][["to"]]
  cor.values = p[["Edgelist"]][["weight"]]
  cor.pval = r_to_p(cor.values, df)
  cor.sig = p_to_sig(cor.pval)
  cor.labels = sprintf("%.2f", cor.values)
  cor.labels = paste0(gsub("-", "\u2013", cor.labels), cor.sig)
  cor = data.frame(var1 = vars[vars.from],
                   var2 = vars[vars.to],
                   cor = cor.values,
                   pval = cor.pval,
                   sig = cor.sig)
  cor = data.frame(cor[order(cor$cor, decreasing=TRUE),],
                   row.names=NULL)
  names(cor)[3] = ifelse(index=="cor", "r", "r.partial")

  if(show.label) {
    edge.label.bg = p[["graphAttributes"]][["Edges"]][["label.bg"]]
    edge.color = p[["graphAttributes"]][["Edges"]][["color"]]
    edge.label.bg[edge.color==edge.color.non] = NA
    p[["graphAttributes"]][["Edges"]][["labels"]] = cor.labels
    p[["graphAttributes"]][["Edges"]][["label.bg"]] = edge.label.bg
  }

  cor.net = list(cor=cor, qgraph=p)
  class(cor.net) = "cor.net"
  attr(cor.net, "plot.params") = list(file = file,
                                      width = width,
                                      height = height,
                                      dpi = dpi)
  return(cor.net)
}


#' @rdname S3method.network
#' @export
plot.cor.net = function(x, scale=1.2, ...) {
  suppressWarnings({
    grob = as_grob(~plot(x$qgraph))
  })
  ggplot() + draw_grob(grob, scale=scale)
}


#' @rdname S3method.network
#' @export
print.cor.net = function(
    x, scale=1.2, file=NULL, width=6, height=4, dpi=500, ...
) {
  if(is.null(file)) {
    plot.params = attr(x, "plot.params")
    file = plot.params$file
    width = plot.params$width
    height = plot.params$height
    dpi = plot.params$dpi
  }

  gg = plot(x, scale)

  index = names(x$cor)[3]  # "r" or "r.partial"
  if(index=="r") {
    algo.text = "Correlation Network"
    file.index = "_COR.NET_correlation.network"
  } else if(index=="r.partial") {
    algo.text = "Partial Correlation Network"
    file.index = "_COR.NET_partial.cor.network"
  } else {
    algo.text = "(Unknown) Correlation Network"
    file.index = "_COR.NET"
  }
  cli::cli_text("Displaying {.pkg {algo.text}}")

  if(is.null(file)) {
    # plot(x$qgraph)  # faster
    print(gg)  # slower
  } else {
    file = file_insert_name(file, file.index)
    ggsave(gg, filename=file, width=width, height=height, dpi=dpi)
    cli::cli_alert_success("Plot saved to {.path {file}}")
  }

  invisible(gg)
}


#' Directed acyclic graphs (DAGs) via Bayesian networks (BNs).
#'
#' Directed acyclic graphs (DAGs) via Bayesian networks (BNs). It uses [bnlearn::boot.strength()] to estimate the strength of each edge as its *empirical frequency* over a set of networks learned from bootstrap samples. It computes (1) the probability of each edge (modulo its direction) and (2) the probabilities of each edge's directions conditional on the edge being present in the graph (in either direction). Stability thresholds are usually set as `0.85` for *strength* (i.e., an edge appearing in more than 85% of BNs bootstrap samples) and `0.50` for *direction* (i.e., a direction appearing in more than 50% of BNs bootstrap samples) (Briganti et al., 2023). Finally, for each chosen algorithm, it returns the stable Bayesian network as the final DAG.
#'
#' @inheritParams cor_net
#' @inheritParams DPI
#' @param algorithm \link[bnlearn:structure-learning]{Structure learning algorithms} for building Bayesian networks (BNs). Should be function name(s) from the [`bnlearn`][bnlearn::bnlearn-package] package. Better to perform BNs with all three classes of algorithms to check the robustness of results (Briganti et al., 2023).
#'
#' Defaults to the most common algorithms: `"pc.stable"` (PC), `"hc"` (HC), and `"rsmax2"` (RS), for the three classes, respectively.
#'
#' - (1) \link[bnlearn:constraint-based algorithms]{Constraint-based Algorithms}
#'   - PC:
#'     \code{"\link[bnlearn:pc.stable]{pc.stable}"}
#'     (*the first practical constraint-based causal structure learning algorithm by Peter & Clark*)
#'   - Others:
#'     \code{"\link[bnlearn:gs]{gs}"},
#'     \code{"\link[bnlearn:iamb]{iamb}"},
#'     \code{"\link[bnlearn:fast.iamb]{fast.iamb}"},
#'     \code{"\link[bnlearn:inter.iamb]{inter.iamb}"},
#'     \code{"\link[bnlearn:iamb.fdr]{iamb.fdr}"}
#' - (2) \link[bnlearn:score-based algorithms]{Score-based Algorithms}
#'   - Hill-Climbing:
#'     \code{"\link[bnlearn:hc]{hc}"}
#'     (*the hill-climbing greedy search algorithm, exploring DAGs by single-edge additions, removals, and reversals, with random restarts to avoid local optima*)
#'   - Others:
#'     \code{"\link[bnlearn:tabu]{tabu}"}
#' - (3) \link[bnlearn:hybrid algorithms]{Hybrid Algorithms} (combination of constraint-based and score-based algorithms)
#'   - Restricted Maximization:
#'     \code{"\link[bnlearn:rsmax2]{rsmax2}"}
#'     (*the general 2-phase restricted maximization algorithm, first restricting the search space and then finding the optimal \[maximizing the score of\] network structure in the restricted space*)
#'   - Others:
#'     \code{"\link[bnlearn:mmhc]{mmhc}"},
#'     \code{"\link[bnlearn:h2pc]{h2pc}"}
#' @param algorithm.args An optional list of extra arguments passed to the algorithm.
#' @param n.boot Number of bootstrap samples (for learning a more "stable" network structure). Defaults to `1000`.
#' @param strength Stability threshold of edge *strength*: the minimum proportion (probability) of BNs (among the `n.boot` bootstrap samples) in which each edge appears.
#' - Defaults to `0.85` (85%).
#' - Two reverse directions share the same edge strength.
#' - Empirical frequency (?~100%) will be mapped onto edge *width/thickness* in the final integrated `DAG`, with wider (thicker) edges showing stronger links, though they usually look similar since the default range has been limited to 0.85~1.
#' @param direction Stability threshold of edge *direction*: the minimum proportion (probability) of BNs (among the `n.boot` bootstrap samples) in which a direction of each edge appears.
#' - Defaults to `0.50` (50%).
#' - The proportions of two reverse directions add up to 100%.
#' - Empirical frequency (?~100%) will be mapped onto edge *greyscale/transparency* in the final integrated `DAG`, with its value shown as edge text label.
#' @param verbose Print information about BN algorithm and number of bootstrap samples when running the analysis. Defaults to `TRUE`.
#'
#' @return
#' Return a list (class `bns.dag`) of Bayesian network results and [`qgraph`][qgraph::qgraph] object.
#'
#' @references
#' Briganti, G., Scutari, M., & McNally, R. J. (2023). A tutorial on Bayesian networks for psychopathology researchers. *Psychological Methods, 28*(4), 947--961. \doi{10.1037/met0000479}
#'
#' Burger, J., Isvoranu, A.-M., Lunansky, G., Haslbeck, J. M. B., Epskamp, S., Hoekstra, R. H. A., Fried, E. I., Borsboom, D., & Blanken, T. F. (2023). Reporting standards for psychological network analyses in cross-sectional data. *Psychological Methods, 28*(4), 806--824. \doi{10.1037/met0000471}
#'
#' Scutari, M., & Denis, J.-B. (2021). *Bayesian networks: With examples in R* (2nd ed.). Chapman and Hall/CRC. \doi{10.1201/9780429347436}
#'
#' <https://www.bnlearn.com/>
#'
#' @seealso
#' [S3method.network]
#'
#' [DPI_dag()]
#'
#' [cor_net()]
#'
#' @examples
#' \donttest{bn = BNs_dag(airquality, seed=1)
#' bn
#' # bn$pc.stable
#' # bn$hc
#' # bn$rsmax2
#'
#' ## All DAG objects can be directly plotted
#' ## or saved with print(..., file="xxx.png")
#' # bn$pc.stable$DAG.edge
#' # bn$pc.stable$DAG.strength
#' # bn$pc.stable$DAG.direction
#' # bn$pc.stable$DAG
#' # ...
#' }
#' \dontrun{
#'
#' print(bn, file="airquality.png")
#' # will save three plots with auto-modified file names:
#' - "airquality_BNs.DAG.01_pc.stable.png"
#' - "airquality_BNs.DAG.02_hc.png"
#' - "airquality_BNs.DAG.03_rsmax2.png"
#'
#' # arrange multiple plots using aplot::plot_list()
#' # install.packages("aplot")
#' c1 = cor_net(airquality, "cor")
#' c2 = cor_net(airquality, "pcor")
#' bn = BNs_dag(airquality, seed=1)
#' mytheme = theme(plot.title=element_text(hjust=0.5))
#' p = aplot::plot_list(
#'   plot(c1),
#'   plot(c2),
#'   plot(bn$pc.stable$DAG) + mytheme,
#'   plot(bn$hc$DAG) + mytheme,
#'   plot(bn$rsmax2$DAG) + mytheme,
#'   design="111222
#'           334455",
#'   tag_levels="A"
#' )  # return a patchwork object
#' ggsave(p, filename="p.png", width=12, height=8, dpi=500)
#' ggsave(p, filename="p.pdf", width=12, height=8)
#' }
#'
#' @export
BNs_dag = function(
    data,
    algorithm = c("pc.stable", "hc", "rsmax2"),
    algorithm.args = list(),
    n.boot = 1000,
    seed = NULL,
    strength = 0.85,
    direction = 0.50,
    node.text.size = 1.2,
    edge.width.max = 1.5,
    edge.label.mrg = 0.01,
    file = NULL,
    width = 6,
    height = 4,
    dpi = 500,
    verbose = TRUE,
    ...
) {
  data = as.data.frame(data)
  vars = names(data)
  for(var in vars) {
    if(is.integer(data[[var]]))
      data[[var]] = as.numeric(data[[var]])
  }

  BNs = lapply(algorithm, function(algo) {
    suppressWarnings({
      if(verbose)
        cli::cli_text("Running BN algorithm {.val {algo}} with {.val {n.boot}} bootstrap samples...")
      set.seed(seed)
      bns = bnlearn::boot.strength(
        data, R=n.boot, algorithm=algo,
        algorithm.args=algorithm.args)
      attr(bns, "algorithm") = algo
      return(bns)
    })
  })
  names(BNs) = algorithm
  gc()

  DAGs = lapply(BNs, function(BN) {
    algo = attr(BN, "algorithm")
    mat.list = bn_to_matrix(BN, strength, direction)
    types = c("edge", "strength", "direction")
    mins = c(edge = 0,
             strength = max(0, strength),
             direction = max(0, direction-0.1))
    maxs = c(edge = 1,
             strength = edge.width.max,
             direction = 1)

    dags = lapply(types, function(type) {
      p = qgraph::qgraph(
        ## --- [data] --- ##
        mat.list[[paste0("mat.", type)]],
        cut = 0,
        minimum = mins[type],
        maximum = maxs[type],

        ## --- [graph] --- ##
        layout = "spring",
        shape = "circle",

        ## --- [label] --- ##
        labels = vars,
        label.cex = node.text.size,
        label.scale.equal = TRUE,

        ## --- [edge] --- ##
        posCol = "black",
        negCol = "red",  # warning (all strength values are in 0~1)
        fade = TRUE,
        edge.labels = ifelse(type=="edge", FALSE, TRUE),
        edge.label.margin = edge.label.mrg,

        ## --- [plotting] --- ##
        usePCH = TRUE,
        DoNotPlot = TRUE,
        ...)

      if(type!="edge") {
        # edge.values = p[["Edgelist"]][["weight"]]
        edge.labels = p[["graphAttributes"]][["Edges"]][["labels"]]
        edge.labels = paste0(100 * as.numeric(edge.labels), "%")
        p[["graphAttributes"]][["Edges"]][["labels"]] = edge.labels
      }

      p$algo = algo
      p$title = paste0(
        "BN algorithm: \"", algo, "\" (DAG value: ", type, ")")
      class(p) = c("bns.dag", class(p))
      return(p)
    })
    names(dags) = paste0("dag.", types)
    return(dags)
  })

  bns.dag = lapply(algorithm, function(algo) {
    bn = BNs[[algo]]
    dag = DAGs[[algo]]
    DAG = DAG.edge = dag[["dag.edge"]]
    DAG.strength = dag[["dag.strength"]]
    DAG.direction = dag[["dag.direction"]]
    DAG[["graphAttributes"]][["Edges"]] =
      DAG.direction[["graphAttributes"]][["Edges"]]
    DAG[["graphAttributes"]][["Edges"]][["width"]] =
      DAG.strength[["graphAttributes"]][["Edges"]][["width"]]
    DAG$title = paste0("BN algorithm: \"", algo, "\"")
    list(
      BN.bootstrap = bn,
      BN = bn[bn$strength > strength & bn$direction > direction,],
      DAG.edge = DAG.edge,
      DAG.strength = DAG.strength,
      DAG.direction = DAG.direction,
      DAG = DAG
    )
  })
  names(bns.dag) = algorithm
  class(bns.dag) = "bns.dag"
  attr(bns.dag, "plot.params") = list(file = file,
                                      width = width,
                                      height = height,
                                      dpi = dpi)
  return(bns.dag)
}


#' @rdname S3method.network
#' @export
plot.bns.dag = function(x, algorithm, scale=1.2, ...) {
  if(inherits(x, "qgraph")) {
    algorithm = x$algo
    title = x$title
  } else {
    # algorithm must be specified
    title = x[[algorithm]][["DAG"]]$title
    x = x[[algorithm]][["DAG"]]
  }
  class(x) = "qgraph"
  suppressWarnings({
    grob = as_grob(~plot(x))
  })
  ggplot() +
    draw_grob(grob, scale=scale) +
    labs(title=title)
}


#' @rdname S3method.network
#' @param algorithm \[For `bns.dag`\] Algorithm(s) to display. Defaults to plot the finally integrated DAG from BN results for each algorithm in `x`.
#' @export
print.bns.dag = function(
    x,
    algorithm = names(x),
    scale = 1.2,
    file=NULL, width=6, height=4, dpi=500,
    ...
) {
  if(inherits(x, "qgraph")) algorithm = x$algo

  if(is.null(file)) {
    plot.params = attr(x, "plot.params")
    file = plot.params$file
    width = plot.params$width
    height = plot.params$height
    dpi = plot.params$dpi
  }

  gg.list = list()

  for(algo in algorithm) {
    gg = plot(x, algo, scale)
    gg.list = c(gg.list, list(gg))

    cli::cli_text("Displaying DAG with BN algorithm {.val {algo}}")

    if(is.null(file)) {
      print(gg)
    } else {
      if(length(algorithm)==1) {
        file.i = file_insert_name(file, sprintf(
          "_BNs.DAG_%s", algo))
      } else {
        file.i = file_insert_name(file, sprintf(
          "_BNs.DAG.%02d_%s", which(algo==algorithm), algo))
      }
      ggsave(gg, filename=file.i, width=width, height=height, dpi=dpi)
      cli::cli_alert_success("Plot saved to {.path {file.i}}")
    }
  }

  names(gg.list) = algorithm
  invisible(gg.list)
}


bn_to_matrix = function(bn, strength=0.85, direction=0.50) {
  bn = bn[bn$strength > strength & bn$direction > direction,]
  vars = attr(bn, "nodes")
  mat.edge = mat.strength = mat.direction =
    matrix(rep(0, length(vars)^2),
           nrow=length(vars),
           dimnames=list(from=vars, to=vars))
  for(i in seq_len(nrow(bn))) {
    mat.edge[bn[i, "from"], bn[i, "to"]] = 1  # logical, replicate
    mat.strength[bn[i, "from"], bn[i, "to"]] = bn[i, "strength"]
    mat.direction[bn[i, "from"], bn[i, "to"]] = bn[i, "direction"]
  }
  return(list(mat.edge = mat.edge,
              mat.strength = mat.strength,
              mat.direction = mat.direction))
}


#' Directed acyclic graphs (DAGs) via DPI exploratory analysis (causal discovery) for all significant partial *r*s.
#'
#' @inheritParams DPI_curve
#' @param data A dataset with at least 3 variables.
#' @param k.covs An integer vector (e.g., `1:10`) of number of random covariates (simulating potential omitted variables) added to each simulation sample. Defaults to `1`. For details, see [DPI()].
#' @param bonf Bonferroni correction to control for false positive rates: `alpha` is divided by, and *p* values are multiplied by, the number of comparisons.
#' - Defaults to `FALSE`: No correction.
#' - `TRUE`: Using the number of all significant partial *r*s.
#' - A user-specified number of comparisons.
#'
#' @return
#' Return a data.frame (class `dpi.dag`) of DPI exploration results.
#'
#' @seealso
#' [S3method.network]
#'
#' [DPI()]
#'
#' [DPI_curve()]
#'
#' [BNs_dag()]
#'
#' [cor_net()]
#'
#' @examples
#' \donttest{# partial correlation networks (undirected)
#' cor_net(airquality, "pcor")
#'
#' # directed acyclic graphs
#' dpi.dag = DPI_dag(airquality, k.covs=c(1,3,5), seed=1)
#' print(dpi.dag, k=1)  # DAG with DPI(k=1)
#' print(dpi.dag, k=3)  # DAG with DPI(k=3)
#' print(dpi.dag, k=5)  # DAG with DPI(k=5)
#'
#' # modify ggplot attributes
#' gg = plot(dpi.dag, k=5, show.label=FALSE)
#' gg + labs(title="DAG with DPI(k=5)")
#'
#' # visualize DPIs of multiple paths
#' ggplot(dpi.dag$DPI, aes(x=k.cov, y=DPI)) +
#'   geom_ribbon(aes(ymin=Sim.LLCI, ymax=Sim.ULCI, fill=path),
#'               alpha=0.1) +
#'   geom_line(aes(color=path), linewidth=0.7) +
#'   geom_point(aes(color=path)) +
#'   geom_hline(yintercept=0, color="red", linetype="dashed") +
#'   scale_y_continuous(limits=c(NA, 0.5)) +
#'   labs(color="Directed Prediction",
#'        fill="Directed Prediction") +
#'   theme_classic()
#' }
#' @export
DPI_dag = function(
    data,
    k.covs = 1,
    n.sim = 1000,
    alpha = 0.05,
    bonf = FALSE,
    seed = NULL,
    progress,
    file = NULL,
    width = 6,
    height = 4,
    dpi = 500
) {
  if(missing(progress)) {
    if(length(k.covs) < 5)
      progress = FALSE
    else
      progress = TRUE
  }

  pcor = cor_net(data, "pcor", edge.width.max=1.5)
  d.pcor = subset(pcor$cor, pval < alpha)[, 1:4]
  n.pcor = nrow(d.pcor)

  if(is.logical(bonf)) bonf = ifelse(bonf, n.pcor, 1)

  cli::cli_text(
    cli::col_cyan("Simulation sample settings: "),
    "k.covs = {cli::col_magenta({k.covs})},
     n.sim = {cli::col_magenta({n.sim})},
     seed = {cli::col_magenta({seed})}")
  cli::cli_text(
    cli::col_cyan("False positive rates (FPR) control: "),
    "Alpha = {cli::col_magenta({format(alpha / bonf, digits=3)})}
     (Bonferroni correction = {cli::col_magenta({bonf})})")

  DPIs = lapply(seq_len(n.pcor), function(i) {
    x = d.pcor[i, 1]
    y = d.pcor[i, 2]
    r.partial = d.pcor[i, 3]
    p.rp = d.pcor[i, 4]
    cli::cli_text(" ")
    cli::cli_text(cli::col_cyan("Exploring [{i}/{n.pcor}]:"))
    cli::cli_text(
      "r.partial =
       {cli::col_yellow({sprintf('%.3f', r.partial)})},
       p = {cli::col_yellow({p.trans(p.rp)})}
       {cli::col_yellow({sig.trans(p.rp)})}")
    DPIs = DPI_curve(x=x, y=y, data=data,
                     k.covs=k.covs, n.sim=n.sim,
                     alpha=alpha, bonf=bonf,
                     seed=seed, progress=progress)
    bonf = attr(DPIs, "bonferroni")
    sign = sign(DPIs[1, "z.value"])
    from = ifelse(sign > 0, x, y)
    to = ifelse(sign > 0, y, x)
    for(j in seq_along(k.covs)) {
      cli::cli_text("
        ---------
        DPI[{.val {from}}->{.val {to}}]({k.covs[j]}) =
        {cli::col_green({sprintf('%.3f', sign * DPIs[j, 'Estimate'])})},
        {ifelse(bonf==1, 'p', paste0('p(Bonf=', bonf, ')'))}
        = {cli::col_green({p.trans(DPIs[j, 'p.z'])})}
        {cli::col_green({sig.trans(DPIs[j, 'p.z'])})}")
    }
    return(data.frame(
      var1 = x,
      var2 = y,
      from = from,
      to = to,
      path = paste(from, to, sep=" \u2192 "),
      r.partial = r.partial,
      p.rp = p.rp,
      k.cov = DPIs$k.cov,
      DPI = sign * DPIs$Estimate,
      Sim.SE = DPIs$Sim.SE,
      z.value = sign * DPIs$z.value,
      p.z = DPIs$p.z,
      Sim.LLCI = sign * DPIs$Sim.LLCI,
      Sim.ULCI = sign * DPIs$Sim.ULCI
    ))
  })

  dpi.dag = list(DPI=do.call("rbind", DPIs), qgraph=pcor$qgraph)
  class(dpi.dag) = c("dpi.dag")
  attr(dpi.dag, "plot.params") = list(file = file,
                                      width = width,
                                      height = height,
                                      dpi = dpi)
  return(dpi.dag)
}


#' @rdname S3method.network
#' @export
plot.dpi.dag = function(
    x,
    k = min(x$DPI$k.cov),
    show.label = TRUE,
    digits.dpi = 2,
    color.dpi.insig = "#EEEEEEEE",
    scale = 1.2,
    ...
) {
  dpi = subset(x$DPI, k.cov==k)
  p = x$qgraph

  vars = p[["Arguments"]][["labels"]]
  vars.from = p[["Edgelist"]][["from"]]
  vars.to = p[["Edgelist"]][["to"]]

  for(i in seq_len(nrow(dpi))) {
    var1 = dpi[i, "var1"]
    var2 = dpi[i, "var2"]
    reverse = var1 == dpi[i, "to"]
    DPI = dpi[i, "DPI"]
    p.z = dpi[i, "p.z"]
    sig = p_to_sig(p.z)
    id = which(vars[vars.from]==var1 & vars[vars.to]==var2)

    if(p.z >= 0.05) {
      # undirected => faded grey edge
      p[["graphAttributes"]][["Edges"]][["color"]][id] = color.dpi.insig
    } else {
      # directed
      p[["Edgelist"]][["directed"]][id] = TRUE
      if(reverse) {
        from.id = p[["Edgelist"]][["from"]][id]
        to.id = p[["Edgelist"]][["to"]][id]
        p[["Edgelist"]][["from"]][id] = to.id
        p[["Edgelist"]][["to"]][id] = from.id
      }
    }
    p[["graphAttributes"]][["Edges"]][["labels"]][id] =
      gsub(
        "0\\.", ".",
        paste0(
          p[["graphAttributes"]][["Edges"]][["labels"]][id],
          "\nDPI.", k, " = ",
          sprintf(paste0("%.", digits.dpi, "f"), DPI),
          sig
        )
      )
  }

  if(show.label==FALSE) {
    p[["graphAttributes"]][["Edges"]][["labels"]] =
      rep(NA, times=length(p[["graphAttributes"]][["Edges"]][["labels"]]))
  }

  suppressWarnings({
    grob = as_grob(~plot(p))
  })
  ggplot() + draw_grob(grob, scale=scale)
}


#' @rdname S3method.network
#' @param k \[For `dpi.dag`\] A single value of `k.cov` to produce the DPI(k) DAG. Defaults to `min(x$DPI$k.cov)`.
#' @param show.label \[For `dpi.dag`\] Show labels of partial correlations, DPI(k), and their significance on edges. Defaults to `TRUE`.
#' @param digits.dpi \[For `dpi.dag`\] Number of decimal places of DPI values displayed on DAG edges. Defaults to `2`.
#' @param color.dpi.insig \[For `dpi.dag`\] Edge color for insignificant DPIs. Defaults to `"#EEEEEEEE"` (faded light grey).
#' @export
print.dpi.dag = function(
    x,
    k = min(x$DPI$k.cov),
    show.label = TRUE,
    digits.dpi = 2,
    color.dpi.insig = "#EEEEEEEE",
    scale = 1.2,
    file=NULL, width=6, height=4, dpi=500,
    ...
) {
  if(is.null(file)) {
    plot.params = attr(x, "plot.params")
    file = plot.params$file
    width = plot.params$width
    height = plot.params$height
    dpi = plot.params$dpi
  }

  gg = plot(x, k, show.label,
            digits.dpi, color.dpi.insig, scale)

  cli::cli_text("Displaying DAG with DPI algorithm (k.cov = {.val {k}})")

  if(is.null(file)) {
    print(gg)
  } else {
    file = file_insert_name(file, "_DPI.DAG")
    ggsave(gg, filename=file, width=width, height=height, dpi=dpi)
    cli::cli_alert_success("Plot saved to {.path {file}}")
  }

  invisible(gg)
}


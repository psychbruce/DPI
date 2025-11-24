# \[S3 methods\] for [`DPI()`](https://psychbruce.github.io/DPI/reference/DPI.md) and [`DPI_curve()`](https://psychbruce.github.io/DPI/reference/DPI_curve.md).

- `summary(dpi)`:

  Summarize DPI results. Return a list (class `summary.dpi`) of
  summarized results and raw DPI data.frame.

- `print(summary.dpi)`:

  Print DPI summary.

- `plot(dpi)`:

  Plot DPI results. Return a `ggplot` object.

- `print(dpi)`:

  Print DPI summary and plot.

- `plot(dpi.curve)`:

  Plot DPI curve analysis results. Return a `ggplot` object.

## Usage

``` r
# S3 method for class 'dpi'
summary(object, ...)

# S3 method for class 'summary.dpi'
print(x, digits = 3, ...)

# S3 method for class 'dpi'
plot(x, file = NULL, width = 6, height = 4, dpi = 500, ...)

# S3 method for class 'dpi'
print(x, digits = 3, ...)

# S3 method for class 'dpi.curve'
plot(x, file = NULL, width = 6, height = 4, dpi = 500, ...)
```

## Arguments

- object:

  Object (class `dpi`) returned from
  [`DPI()`](https://psychbruce.github.io/DPI/reference/DPI.md).

- ...:

  Other arguments (currently not used).

- x:

  Object (class `dpi` or `dpi.curve`) returned from
  [`DPI()`](https://psychbruce.github.io/DPI/reference/DPI.md) or
  [`DPI_curve()`](https://psychbruce.github.io/DPI/reference/DPI_curve.md).

- digits:

  Number of decimal places. Defaults to `3`.

- file:

  File name of saved plot (`".png"` or `".pdf"`).

- width, height:

  Width and height (in inches) of saved plot. Defaults to `6` and `4`.

- dpi:

  Dots per inch (figure resolution). Defaults to `500`.

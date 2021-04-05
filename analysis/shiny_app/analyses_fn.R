library(ggplot2)
library(devtools)
library(ggpubr)
source_gist("524eade46135f6348140")

makeggplot <- function(dataP, xP, yP, colorP, statCorFn = NULL) {
  ggplot(dataP, aes_string(xP , yP, color=colorP)) +
    stat_smooth_func(geom="text", method="lm", hjust=0, parse=TRUE) +
    geom_smooth(method="lm",se=FALSE) +
    geom_point(alpha = 0.3) + statCorFn
}
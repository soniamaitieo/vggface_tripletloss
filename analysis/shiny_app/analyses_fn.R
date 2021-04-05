library(ggplot2)
library(devtools)
library(ggpubr)
library(FactoMineR)
library(plotly)
source_gist("524eade46135f6348140")

makeggplot <- function(dataP, xP, yP, colorP, statCorFn = NULL) {
  ggplot(dataP, aes_string(xP , yP, color=colorP)) +
    stat_smooth_func(geom="text", method="lm", hjust=0, parse=TRUE) +
    geom_smooth(method="lm",se=FALSE) +
    geom_point(alpha = 0.3) + statCorFn
}

makepca <- function(CFD, dim1 = 1, dim2 = 2) {
  CFDquant <- CFD[,5:12]
  CFDquant <-CFD[ , purrr::map_lgl(CFD, is.numeric)]
  pca.CFD <- PCA((CFDquant[ , colSums(is.na(CFDquant)) == 0]))
  CFDquant_withoutAF <-CFDquant[ , -which(names(CFDquant) %in% c("Feminine","Attractive"))]
  pca.CFD2 <- PCA((CFDquant_withoutAF [ , colSums(is.na(CFDquant_withoutAF )) == 0]),graph=F)

  p <- plot_ly(CFD,x=pca.CFD2$ind$coord[,dim1] ,y=pca.CFD2$ind$coord[,dim2],text=CFD$Model,
               mode="markers",color = CFD$GenderSelf,marker=list(size=11))
  p <- layout(p,title="PCA",
              xaxis=list(title= paste("PC" ,dim1, ": Variance ",round(pca.CFD2$eig[dim1,2]),"%")),
              yaxis=list(title= paste("PC" ,dim2, ": Variance ",round(pca.CFD2$eig[dim2,2]),"%")))
  p
}

#res_emb <- read.delim("~/vggface_tripletloss/results/2021-03-03-15-33/CFDvecs_testCFD.tsv", header=FALSE)
#res_emb <- read.delim("/home/sonia/vggface_tripletloss/results/2021-03-04-17-13/CFDvecs_N.tsv", header=FALSE)
#res_emb <- read.delim("/home/sonia/vggface_tripletloss/results/2021-03-04-17-13/CFDvecsall.tsv", header=FALSE)


#------------------------------------------------------------------------------#
#Individual Verification
res_emb1 <- read.delim("/home/sonia/vggface_tripletloss/results/2021-03-12-8-47/CFDvecs.tsv", header=FALSE)
#Gender Verification
res_emb2 <- read.delim("/home/sonia/vggface_tripletloss/results/2021-03-17-17-16/CFDvecs.tsv", header=FALSE)
#Individual Classification
res_emb3 <- read.delim("/home/sonia/vggface_tripletloss/results/2021-04-15-11-46/CFDvecs.tsv", header=FALSE)
#Gender Classification
res_emb4 <- read.delim("/home/sonia/vggface_tripletloss/results/2021-04-19-11-10/CFDvecs.tsv", header=FALSE)

CFD1 <- read.csv("~/vggface_tripletloss/results/2021-03-12-8-47/CFD_N_analysis.csv", row.names=1)
CFD2 <- read.csv("~/vggface_tripletloss/results/2021-03-17-17-16/CFD_N_analysis.csv", row.names=1)
CFD3 <- read.table("/home/sonia/vggface_tripletloss/results/2021-04-15-11-46/CFD_N_analysis.csv", quote="\"", comment.char="" , sep = "," , header = T)
CFD4 <- read.table("/home/sonia/vggface_tripletloss/results/2021-04-19-11-10/CFD_N_analysis.csv", quote="\"", comment.char="" , sep = "," , header = T)


library(FactoMineR)
library(plotly)
library(e1071)



getPCS <- function(res_emb){
  dt.acp=PCA(res_emb ,axes = c(1,2),ncp=30)
  head(dt.acp$eig)
  #variabilité expliqés par les différents composants - distance cum des variabilités
  barplot(dt.acp$eig[,1],main="Eboulis de valeur propre", xlab="Nb de composants", names.arg=1:nrow(dt.acp$eig))
  #dt.acp.30dim = dt.acp$ind$coord
  print(paste("Cumulative percentage of variance for 30 PCs",round(dt.acp$eig[30,3]),"%"))
  return(dt.acp)
}

  
plotlyPCA <- function(dim1,dim2,dt.acp.30dim,CFD){
  #variables à choisir par shiny
  dim1 = 1
  dim2 = 3
  p <- plot_ly(CFD,x=dt.acp$ind$coord[,dim1] ,y=dt.acp$ind$coord[,dim2],text=CFD$Model,
               mode="markers",color = CFD$GenderSelf,marker=list(size=11))
  p <- layout(p,title="PCA",
              xaxis=list(title= paste("PC" ,dim1, ": Variance ",round(dt.acp$eig[dim1,2]),"%")),
              yaxis=list(title= paste("PC" ,dim2, ": Variance ",round(dt.acp$eig[dim2,2]),"%")))
  
  p
  }

plotlyPCA(dim1=1,dim2=2,dt.acp,CFD)

dt.acp1 <- getPCS(res_emb1)
dt.acp2 <- getPCS(res_emb2)
dt.acp3 <- getPCS(res_emb3)
dt.acp4 <- getPCS(res_emb4)


calcCentroidPCA <- function(dt.acp,CFD){
  c.acp = apply(dt.acp$ind$coord[,1:30],2,mean)
  dc.acp = apply(dt.acp$ind$coord[,1:30], 1, FUN = function(x) sqrt(sum((x-c.acp)^2)))
  cf.acp = apply(dt.acp$ind$coord[which(CFD$GenderSelf == "F"),1:30],2,mean) 
  dcf.acp = apply(dt.acp$ind$coord[,1:30], 1, FUN = function(x) sqrt(sum((x-cf.acp)^2)))
  cm.acp = apply(dt.acp$ind$coord[which(CFD$GenderSelf == "M"),1:30],2,mean) 
  dcm.acp = apply(dt.acp$ind$coord[,1:30], 1, FUN = function(x) sqrt(sum((x-cm.acp)^2)))
  CFD=cbind(CFD,dc.acp ,dcf.acp,dcm.acp)
  return(CFD)
}

CFD1 <- calcCentroidPCA(dt.acp1,CFD1)
CFD2 <- calcCentroidPCA(dt.acp2,CFD2)
CFD3 <- calcCentroidPCA(dt.acp3,CFD3)
CFD4 <- calcCentroidPCA(dt.acp4,CFD4)


calcPfPCA <- function(dt.acp,CFD){
  modelsvm <- svm(dt.acp$ind$coord[,1:30], CFD$GenderSelf, probability=TRUE) 
  summary(modelsvm)
  # test with train data
  pred <- predict(modelsvm, dt.acp$ind$coord[,1:30], probability=TRUE)
  # Check accuracy:
  print(table(pred, CFD$GenderSelf))
  pFsvm.pca = attr(pred, "probabilities")[,'F']
  CFD$pFsvm.pca = pFsvm.pca
  return(CFD)
}

CFD1 <- calcPfPCA(dt.acp1,CFD1)
CFD2 <- calcPfPCA(dt.acp2,CFD2)
CFD3 <- calcPfPCA(dt.acp3,CFD3)
CFD4 <- calcPfPCA(dt.acp4,CFD4)


ggplot(data = CFD, aes(x = Attractive , y =dcf.acp, color=GenderSelf)) +
  #stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) + stat_cor(aes(color = GenderSelf))


ggplot(data = CFD, aes(x = Attractive , y =dc.acp)) +
  #stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) + stat_cor()


CFD$ratioFH <- CFD$dist_centre_F / CFD$dist_centre_H

ggplot(data = CFD, aes(x = Feminine , y = CFD$ratioFH, color=GenderSelf)) +
  #stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) + stat_cor(aes(color = GenderSelf))


ggplot(data = CFD, aes(x = Feminine , y = CFD$dist_centre_F)) +
  #stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) + stat_cor()


ggplot(data = CFD4, aes(x = pF_svm_linear , y = pFsvm.pca, color=GenderSelf)) +
  #stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) + stat_cor(aes(color = GenderSelf))


ggplot(data = CFD2, aes(x = rank(pF_svm_linear) , y = rank(pFsvm.pca))) +
  #stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) + stat_cor()


ggplot(data = CFD4, aes(x = dist_centre_H , y = dcm.acp)) +
  #stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) + stat_cor()


CFD<- read.csv("~/vggface_tripletloss/results/2021-03-04-17-13/CFD_N_analysis.csv", row.names=1)
CFD<- read.csv("~/vggface_tripletloss/results/2021-03-12-8-47/CFD_N_analysis.csv", row.names=1)
CFD<- read.csv("~/vggface_tripletloss/results/2021-03-17-17-16/CFD_N_analysis.csv", row.names=1)

library(ggplot2)
library(devtools)
library(ggpubr)
source_gist("524eade46135f6348140")

ggplot(data = CFD, aes(x = Attractive , y =LL)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3)


ggplot(data = CFD, aes(x = Attractive , y =rank(LL))) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3)

ggplot(data = CFD, aes(x = Attractive , y =LL_F)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3)


ggplot(data = CFD[which(CFD$GenderSelf == "F"),], aes(x = Attractive , y =rank(LL_F))) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3)


ggplot(data = CFD, aes(x = Attractive , y =LL_M)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) +  stat_cor()


ggplot(data = CFD[which(CFD$GenderSelf == "F"),], aes(x = Attractive , y =LL_F)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) +    stat_cor()


ggplot(data = CFD, aes(x = Feminine , y =Attractive, color=GenderSelf)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3)  +  stat_cor(aes(color = GenderSelf))

geom_point(alpha = 0.3) +    stat_cor()


ggplot(data = CFD, aes(x = Feminine , y =rank(dist_centre_F), color=GenderSelf))) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3)  +  stat_cor(aes(color = GenderSelf))




ggplot(data = CFD, aes(x = Feminine , y =rank(dist_centre_F), color=GenderSelf)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) +  stat_cor(aes(color = GenderSelf))


ggplot(data = CFD, aes(x = Attractive , y =rank(dist_centre_H), color=GenderSelf)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3)  +  stat_cor(aes(color = GenderSelf))

ggplot(data = CFD, aes(x = Attractive , y =LL, color=GenderSelf)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3)  +  stat_cor(aes(color = GenderSelf))



ggplot(data = CFD, aes(x = Feminine , y =rank(proba_F_svm), color=GenderSelf)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3)


ggplot(data = CFD, aes(x = Feminine , y =rank(pF_svm_linear), color=GenderSelf)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3)


ggplot(data = CFD, aes(x = Feminine , y =rank(pF_svm_linear), color=GenderSelf)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3)  +  stat_cor(aes(color = GenderSelf))



ggplot(data = CFD, aes(x = Feminine , y =pF_svm_rbf, color=GenderSelf)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) +  stat_cor(aes(color = GenderSelf))

ggplot(data = CFD, aes(x = Attractive, y =rank(pF_svm_linear))) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) + stat_cor()


ggplot(data = CFD, aes(x = Attractive , y =rank(pF_svm_rbf), color=GenderSelf)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) +  stat_cor(aes(color = GenderSelf))

sp <- ggscatter(CFD, x = "Attractive", y = "LL",
                palette = "jco",
                add = "reg.line", conf.int = TRUE)
sp + stat_cor()
#> `geom_smooth()` using formula 'y ~ x'


ggplot(data = df1, aes(x = mean_umap1 , y =LL , color=GenderSelf)) +
  stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) +
  geom_point(alpha = 0.3) +  stat_cor(aes(color = GenderSelf))


#--


res_emb <- read.delim("/home/sonia/vggface_tripletloss/results/2021-03-12-8-47/CFDvecs.tsv", header=FALSE)
CFD<- read.csv("~/vggface_tripletloss/results/2021-03-12-8-47/CFD_N_analysis.csv", row.names=1)

res_emb <- read.delim("/home/sonia/vggface_tripletloss/results/2021-03-17-17-16/CFDvecs.tsv", header=FALSE)
CFD <- read.table("/home/sonia/vggface_tripletloss/results/2021-03-17-17-16/CFD_N_analysis.csv", quote="\"", comment.char="", sep = "," , header = T)


centroide = sapply(res_emb,mean)
dist_centre = apply(res_emb, 1, FUN = function(x) sqrt(sum((x-centroide)^2)) )
CFD$dist_centroid <- dist_centre


ggplot(data = CFD, aes(x = Attractive, y = pF_svm_linear, color=GenderSelf)) +
  #stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) + 
  geom_point(alpha = 0.3) +  theme_bw() + stat_cor(aes(color = GenderSelf),method = "pearson",size=5)  

ggplot(data = CFD, aes(x = Attractive, y = dist_centre_H, color=GenderSelf)) +
  #stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) + 
  geom_point(alpha = 0.3) +  theme_bw() + 
  stat_cor(aes(color = GenderSelf),method = "pearson",size=5, label.y.npc="middle",label.x.npc="middle")  


ggplot(data = CFD, aes(x = Attractive, y = dist_centre_F, color=GenderSelf)) +
  #stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) + 
  geom_point(alpha = 0.3) +  theme_bw() + stat_cor(aes(color = GenderSelf),method = "pearson",size=5)  




ggplot(data = CFD, aes(x = Attractive, y = rank(pF_svm_linear), color=GenderSelf)) +
  #stat_smooth_func(geom="text",method="lm",hjust=0,parse=TRUE) +
  geom_smooth(method="lm",se=FALSE) + 
  geom_point(alpha = 0.3) +  theme_bw() + stat_cor(aes(color = GenderSelf))  




pos_F = which(CFDlabel$GenderSelf == "M")
res_emb_F = res_emb[pos_F,]
meanF = sapply(res_emb_F,mean)
dist_centroid_Fem = apply(res_emb, 1, FUN = function(x) sqrt(sum((x-meanF)^2)) )
CFDlabel$dist_centroid_Fem <- dist_centroid_Fem

pval <- c()
for (i in 1:length(res_emb_F)){
  print(i)
  #hist(res_emb_F[,i])
  pval=append(pval,shapiro.test(res_emb_F[,i])$p.value)
  #break
}

sum(pval > 0.05)


pval <- c()
for (i in 1:length(res_emb)){
    print(i)
  hist(res_emb_F[,i])
  pval=append(pval,shapiro.test(res_emb[,i])$p.value)
  break
}
length(which(pval > 0.05) )


U <- function(mu,y) {
  e1 <- (y-mu[1])^2
  e2 <- (y-mu[2])^2
  e.min <- pmin(e1,e2)
  e <- sum(e.min)
  return(e)
}


y=res_emb[,i]
r.cluster <- nlm(U,c(50,80),y)
mu.est <- r.cluster$estimate
print(mu.est)


r.km <- kmeans(y, centers=2)
r.km$cluster
as.vector(r.km$centers)



r.nlm <- nlm(mixt.deviance,c(.25,52,82,10,10),y)
theta.est <- c(r.nlm$estimate[1], 1-r.nlm$estimate[1], r.nlm$estimate[2:5])
print(matrix(theta.est,nrow=3,byrow = T))



library(FactoMineR)
CFDquant <- CFD[,5:12]
CFDquant <-CFD[ , purrr::map_lgl(CFD, is.numeric)]
pca.CFD <- PCA((CFDquant[ , colSums(is.na(CFDquant)) == 0]))
CFDquant_withoutAF <-CFDquant[ , -which(names(CFDquant) %in% c("Feminine","Attractive"))]
pca.CFD2 <- PCA((CFDquant_withoutAF [ , colSums(is.na(CFDquant_withoutAF )) == 0]))


library(plotly)

#variables à choisir par shiny
dim1 = 1
dim2 = 2
p <- plot_ly(CFD,x=pca.CFD2$ind$coord[,dim1] ,y=pca.CFD2$ind$coord[,dim2],text=CFD$Model,
             mode="markers",color = CFD$GenderSelf,marker=list(size=11))
p <- layout(p,title="PCA",
            xaxis=list(title= paste("PC" ,dim1, ": Variance ",round(pca.CFD2$eig[dim1,2]),"%")),
            yaxis=list(title= paste("PC" ,dim2, ": Variance ",round(pca.CFD2$eig[dim2,2]),"%")))

p






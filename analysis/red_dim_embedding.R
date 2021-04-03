res_emb <- read.delim("~/vggface_tripletloss/results/2021-03-03-15-33/CFDvecs_testCFD.tsv", header=FALSE)
#res_emb <- read.delim("/home/sonia/vggface_tripletloss/results/2021-03-04-17-13/CFDvecs_N.tsv", header=FALSE)
#res_emb <- read.delim("/home/sonia/vggface_tripletloss/results/2021-03-04-17-13/CFDvecsall.tsv", header=FALSE)

library(readxl)
CFD<- read_excel("cfd/CFDVersion2.5/CFD2.5NormingDataandCodebook.xlsx", 
                                           sheet = "CFD_U.S._NormingData", skip = 7)


CFDlabel <- read.table("~/vggface_tripletloss/results/2021-03-03-15-33/CFDlabel_testCFD.tsv", quote="\"", comment.char="")
#CFDlabel <- read.table("/home/sonia/vggface_tripletloss/results/2021-03-04-17-13/CFDlabelall.tsv", quote="\"", comment.char="")


CFD$Model<-as.factor(CFD$Model)
CFD_Model <- read.table("~/vggface_tripletloss/results/2021-03-04-17-13/CFD_Model.tsv", quote="\"", comment.char="")


lev = levels(as.factor(CFD$Model))
label = factor(CFDlabel$V1, labels = lev)


#only_multi = 110:1150
#emb_multi = res_emb[only_multi,]
#label_multi = as.factor(label[only_multi])
#ethn = as.factor(substr(label_multi, start = 1, stop = 1))
#gender = as.factor(substr(label_multi, start = 2, stop = 2))

ethn = as.factor(substr(label, start = 1, stop = 1))
write.table(ethn, file='/home/sonia/vggface_tripletloss/results/2021-03-03-15-33/CFDlabelall_ethn.tsv', quote=FALSE, sep='\t', col.names = NA)
gender = as.factor(substr(label, start = 2, stop = 2))
write.table(gender, file='/home/sonia/vggface_tripletloss/results/2021-03-03-15-33/CFDlabelall_gender.tsv', quote=FALSE, sep='\t', col.names = NA)

emb_multi = res_emb
library(umap)
#emb.umap = umap(emb_multi,method = "umap-learn")
emb.umap = umap(emb_multi)
umap(emb_multi,labels=label,controlscale=TRUE,scale=3)

df_out=data.frame(umap1=emb.umap$layout[,3], umap2=emb.umap$layout[,4], indiv=label, ethn=ethn,gender=gender) 

library(ggplot2)

df_out=data.frame(umap1=emb.umap$data[,1], umap2=emb.umap$data[,2], indiv=label_multi, ethn=ethn,gender=gender) 

df_out=data.frame(umap1=emb.umap$data[,1], umap2=emb.umap$data[,2], indiv=label, ethn=ethn,gender=gender) 

p<-ggplot(df_out,aes(x=umap1,y=umap2,color=gender))
p<-p+geom_point()  
p

# Diagramme de dispersion
p2 <- ggplot(df_out,aes(x=umap1,y=umap2))+
  geom_point(size=2,aes(color = indiv, shape = gender)) + 
    stat_ellipse(aes(x=umap1,y=umap2,color=ethn, group=ethn)) +  
  scale_colour_manual(values =rainbow(length(unique(label))+10))  + guides(color = FALSE)
p2 



hist(emb.umap$data[,1], prob=TRUE, col="grey")# prob=TRUE for probabilities not counts
lines(density(emb.umap$data[,1]), col="blue", lwd=2) # add a density estimate with defaults
lines(density(emb.umap$data[,1], adjust=2), lty="dotted", col="darkgreen", lwd=2) 

lapply(emb.umap$data , shapiro.test)
shapiro.test(emb.umap$layout[,1])$p.value

df.shapiro <- apply(res_emb, 2, shapiro.test)
L = unlist(lapply(df.shapiro, function(x) x$p.value))

#-------------------------

label= as.factor(label)
ethn = as.factor(substr(label, start = 1, stop = 1))
gender = as.factor(substr(label, start = 2, stop = 2))

labelF = label[gender == "F"]
ethnF =  ethn[gender == "F"]
res_embF = res_emb[ label[gender == "F"],]

hist(emb.umap$data[,1], prob=TRUE, col="grey")# prob=TRUE for probabilities not counts
lines(density(emb.umap$data[,1]), col="blue", lwd=2) # add a density estimate with defaults
lines(density(emb.umap$data[,1], adjust=2), lty="dotted", col="darkgreen", lwd=2) 

Sigma <- var(res_emb)
Means <- colMeans(res_emb)
simulation <- mvrnorm(n = 1000, Means, Sigma)


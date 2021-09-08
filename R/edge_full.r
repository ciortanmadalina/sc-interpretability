# http://www.nathalievialaneix.eu/doc/html/solution_edgeR-tomato-withcode.html
#source("http://bioconductor.org/biocLite.R")
#BiocManager::install("http://bioconductor.org/biocLite.R")
#biocLite("edgeR")
library(edgeR)
require(rhdf5)
setwd("/home/rstudio/projects/interpretability/R")
#library(DEsingle)

analyze<-function(counts, label, datasetname, method, category){
  ifelse (!dir.exists(paste0("interpretability/", category)), 
          dir.create(paste0("interpretability/", category)), FALSE)
  ifelse (!dir.exists(paste0("interpretability/", category, '/', method)), 
          dir.create(paste0("interpretability/", category, '/', method)), FALSE)
  ifelse (!dir.exists(paste0("interpretability/", category, '/', method, '/',datasetname)), 
          dir.create(paste0("interpretability/", category, '/', method,'/', datasetname)), FALSE)
  
  for (val in unique(label)){

    start_time <- Sys.time()
    vec <- factor(as.integer(label ==val))
    y <- DGEList(counts=datacount,group=vec)
    keep <- filterByExpr(y)
    y <- y[keep,,keep.lib.sizes=FALSE]
    y <- calcNormFactors(y)
    design <- model.matrix(~vec)
    y <- estimateDisp(y,design)
    
    fit <- glmQLFit(y,design)
    qlf <- glmQLFTest(fit,coef=2)
    result<-qlf$table
    end_time <- Sys.time()
    result["time"] = as.numeric(difftime(end_time, start_time, tz="GMT" , units="secs"))

    write.csv(result, paste0("interpretability/",category, '/', method, '/', datasetname, "/edgerfull_", val, ".csv"))

  }
}
get_input_data<-function(data_path){
  h5closeAll()
  datacount <- h5read(file, "X")
  print(paste0(file, " nrows ", nrow(datacount), " ncols ", ncol(datacount)))
  cell_label<- h5read(file, "Y")
  colnames(datacount) <-seq(1, ncol(datacount), length.out=ncol(datacount))
  rownames(datacount) <-seq(1, nrow(datacount), length.out=nrow(datacount))
  
  print(nrow(datacount)) # 23000
  print(ncol(datacount)) # 3660
  return(list(datacount, cell_label))
}  
# Run on simulated
datasetnames <-c('data_1c_3de_0.3',
                 'data_0c_3de_0.1',
                 'data_0c_2de_0.05',
                 'data_-1c_3de_0.05',
                 'data_1c_3de_0.05',
                 'data_1c_2de_0.3',
                 'data_-1c_3de_0.3',
                 'data_1c_2de_0.05',
                 'data_-1c_2de_0.3',
                 'data_1c_3de_0.1',
                 'data_0c_3de_0.3',
                 'data_1c_2de_0.1',
                 'data_-1c_2de_0.1',
                 'data_0c_3de_0.05',
                 'data_-1c_3de_0.1',
                 'data_0c_2de_0.1',
                 'data_-1c_2de_0.05',
                 'data_0c_2de_0.3')
datasetname <-"data_-1c_3de_0.1"
threshold <-0.99
category <- 'balanced_data'
methods <- c("contrastivesc", "scDeepCluster", "scziDesk", "truth")
method <- "contrastivesc"
file = paste0("simulated_data/balanced_data/", datasetname, ".h5")
preds <- read.csv(file = paste0('../output/interpretability/', category, '/', method, '/', datasetname, '.csv'))
idxdf <- read.csv(file = paste0('../output/interpretability/', category, '/', method, '/', datasetname, '_selected.csv'))
output = get_input_data(file)
datacount = output[[1]]
print(nrow(datacount)) #
print(ncol(datacount)) # 
#idx = as.vector(unlist(idxdf["idx"]))
#datacount = t(datacount)
#datacount = datacount[, colnames(datacount)[idx]]
#colnames(datacount) <-seq(0, ncol(datacount)-1)
#datacount <-t(datacount)

# test
for (methodname in colnames(preds)[-c(1)]){
  print(methodname)
  label <-as.vector(unlist(preds[methodname]))
  analyze(datacount, label, datasetname, methodname, category)
}
categories <- c('imbalanced_data','balanced_data')
for (q in 1: length(categories)){
  category <- categories[q]
  for(d in 1:length(datasetnames)){
    datasetname <-datasetnames[d]
    for(m in 1:length(methods)){
      method <- methods[m]
      file = paste0("simulated_data/", category, "/", datasetname, ".h5")
      preds <- read.csv(file = paste0('../output/interpretability/', category, '/', method, '/', datasetname, '.csv'))
      idxdf <- read.csv(file = paste0('../output/interpretability/', category, '/', method, '/', datasetname, '_selected.csv'))
      output = get_input_data(file)
      datacount = output[[1]]
      cell_label =output[[2]]
      print(nrow(datacount)) #
      print(ncol(datacount)) # 
      #analyze(datacount, label, datasetname, methodname, threshold)
      
      for (methodname in colnames(preds)[-c(1)]){
        print(methodname)
        label <-as.vector(unlist(preds[methodname]))
        analyze(datacount, label, datasetname, methodname, category)
        break()
      }
    }
  }
}
# Run on scRNA-seq
datasetname <-"Quake_Smart-seq2_Trachea"
threshold <-0.05
preds <- read.csv(file = paste0('../output/interpretability/', category, '/', method, '/',datasetname, '.csv'))
idxdf <- read.csv(file = paste0('../output/interpretability/', category, '/', method, '/',datasetname, '_selected.csv'))
file = paste0("../real_data/", datasetname, ".h5")
output = get_input_data(file)
datacount = output[[1]]
print(nrow(datacount)) #
print(ncol(datacount)) # 
idx = as.vector(unlist(idxdf["idx"]))
datacount = t(datacount)
datacount = datacount[, colnames(datacount)[idx]]
colnames(datacount) <-seq(0, ncol(datacount)-1)
datacount <-t(datacount)


# test
for (methodname in colnames(preds)[-c(1)]){
  print(methodname)
  label <-as.vector(unlist(preds[methodname]))
  analyze(datacount, label, datasetname, methodname, category)
}



methodname = "contrastivesc"
label <-as.vector(unlist(preds[methodname]))
print(nrow(datacount)) # 
print(ncol(datacount)) # 
for (val in unique(label)){
  start_time <- Sys.time()
  vec <- factor(as.integer(label ==val))
}


y <- DGEList(counts=datacount,group=vec)
keep <- filterByExpr(y)
y <- y[keep,,keep.lib.sizes=FALSE]
y <- calcNormFactors(y)
design <- model.matrix(~vec)
y <- estimateDisp(y,design)
# To perform quasi-likelihood F-tests:
fit <- glmQLFit(y,design)
qlf <- glmQLFTest(fit,coef=2)
result<-qlf$table
result["time"] = 3







result<-topTags(qlf)

#o perform likelihood ratio tests:
fit <- glmFit(y,design)
lrt <- glmLRT(fit,coef=2)
lrt$table
topTags(lrt)




data(TestData)
dim(counts)
group <- factor(c(rep(1,50), rep(2,100)))
group <- factor(c(rep(1,50), rep(2,70), rep(3,30)))
# manual https://bioconductor.org/packages/release/bioc/vignettes/edgeR/inst/doc/edgeRUsersGuide.pdf

y <- DGEList(counts=counts,group=group)
keep <- filterByExpr(y)
y <- y[keep,,keep.lib.sizes=FALSE]
y <- calcNormFactors(y)
design <- model.matrix(~group)
y <- estimateDisp(y,design)
# To perform quasi-likelihood F-tests:
fit <- glmQLFit(y,design)
qlf <- glmQLFTest(fit,coef=2)
topTags(qlf)
#o perform likelihood ratio tests:
fit <- glmFit(y,design)
lrt <- glmLRT(fit,coef=2)
topTags(lrt)

# tutorial http://www.nathalievialaneix.eu/doc/html/solution_edgeR-tomato-withcode.html
dgeFull <- DGEList(counts, group=group)

dgeFull <- DGEList(dgeFull$counts[apply(dgeFull$counts, 1, sum) != 0, ],
                   group=dgeFull$samples$group)
head(dgeFull$counts)

dgeFull <- calcNormFactors(dgeFull, method="TMM")
dgeFull$samples

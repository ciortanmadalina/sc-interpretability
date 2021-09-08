#https://miaozhun.github.io/DEsingle/
#if (!requireNamespace("BiocManager", quietly=TRUE))
#  install.packages("BiocManager")
#BiocManager::install("DEsingle")

library(DEsingle)
require(rhdf5)
setwd("/home/rstudio/projects/interpretability/R")
category <- ""
ifelse (!dir.exists(paste0("interpretability/", category)), 
        dir.create(paste0("interpretability/", category)), FALSE)

analyze<-function(counts, label, datasetname, method, threshold, category){
  ifelse (!dir.exists(paste0("interpretability/", category)), 
          dir.create(paste0("interpretability/", category)), FALSE)
  ifelse (!dir.exists(paste0("interpretability/", category, '/', method)), 
          dir.create(paste0("interpretability/", category, '/', method)), FALSE)
  ifelse (!dir.exists(paste0("interpretability/", category, '/', method, '/',datasetname)), 
          dir.create(paste0("interpretability/", category, '/', method,'/', datasetname)), FALSE)
  for (val in unique(label)){
    start_time <- Sys.time()
    vec <- factor(as.integer(label ==val))
    results <- DEsingle(counts = counts, group = vec)
    results.classified <- DEtype(results = results, threshold = threshold)
    results.sig <- results.classified[results.classified$pvalue.adj.FDR < threshold, ]
    end_time <- Sys.time()
    results$time = as.numeric(difftime(end_time, start_time, tz="GMT" , units="secs"))

    write.csv(results, paste0("interpretability/",category, '/', method, '/', datasetname, "/desingle_", val, ".csv"))
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
                 'data_0c_2de_0.3'
                 )

datasetnames <- rev(datasetnames)
# Run on simulated
datasetname <-"data_-1c_3de_0.1"
threshold <-0.1
category <- 'imbalanced_data'
methods <- c("scziDesk", "truth", "contrastivesc", "scDeepCluster")
method <- "contrastivesc"
file = paste0("simulated_data/", category, "/", datasetname, ".h5")
preds <- read.csv(file = paste0('../output/interpretability/', category, '/', method, '/', datasetname, '.csv'))
idxdf <- read.csv(file = paste0('../output/interpretability/', category, '/', method, '/', datasetname, '_selected.csv'))
output = get_input_data(file)
datacount = output[[1]]
cell_label =output[[2]]
print(nrow(datacount)) #
print(ncol(datacount)) # 
idx = as.vector(unlist(idxdf["idx"]))
datacount = t(datacount)
datacount = datacount[, colnames(datacount)[idx]]
colnames(datacount) <-seq(0, ncol(datacount)-1)
datacount <-t(datacount)
print(nrow(datacount)) # 
print(ncol(datacount)) # 
#analyze(datacount, label, datasetname, methodname, threshold)
colnames(preds)
for (methodname in colnames(preds)[-c(1)]){
  print(methodname)
  break()
  #label <-as.vector(unlist(preds[methodname]))
  #analyze(datacount, label, datasetname, methodname, threshold, category)
}

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
    idx = as.vector(unlist(idxdf["idx"]))
    datacount = t(datacount)
    datacount = datacount[, colnames(datacount)[idx]]
    colnames(datacount) <-seq(0, ncol(datacount)-1)
    datacount <-t(datacount)
    print(nrow(datacount)) # 
    print(ncol(datacount)) # 
    #analyze(datacount, label, datasetname, methodname, threshold)
    
    for (methodname in colnames(preds)[-c(1)]){
      print(methodname)
      label <-as.vector(unlist(preds[methodname]))
      analyze(datacount, label, datasetname, methodname, threshold, category)
      break()
    }
  }
}
#simulated_data/balanced_data/data_1c_2de_0.3.h5,
#simulated_data/imbalanced_data/data_0c_3de_0.1.h5
#simulated_data/imbalanced_data/data_-1c_2de_0.05.h5
#simulated_data/balanced_data/data_0c_3de_0.3.h5
# Run on scRNA-seq
datasetname <-"Quake_Smart-seq2_Trachea"
threshold <-0.05
preds <- read.csv(file = paste0('../output/interpretability/', datasetname, '.csv'))
idxdf <- read.csv(file = paste0('../output/interpretability/', datasetname, '_selected.csv'))
file = paste0("../real_data/", datasetname, ".h5")
output = get_input_data(file)
datacount = output[[1]]
cell_label =output[[2]]
print(nrow(datacount)) #
print(ncol(datacount)) # 
idx = as.vector(unlist(idxdf["idx"]))
datacount = t(datacount)
datacount = datacount[, colnames(datacount)[idx]]
colnames(datacount) <-seq(0, ncol(datacount)-1)
datacount <-t(datacount)
label <-as.vector(unlist(preds[methodname]))
print(nrow(datacount)) # 
print(ncol(datacount)) # 
#analyze(datacount, label, datasetname, methodname, threshold)

for (methodname in colnames(preds)[-c(1)]){
  print(methodname)
  label <-as.vector(unlist(preds[methodname]))
  analyze(datacount, label, datasetname, methodname, threshold, category)
}


# TEST
data(TestData)
label <- factor(c(rep(1,50), rep(2,25),  rep(3,25),  rep(4,50)))
datasetname <-"test"
methodname <-"ground_truth"
threshold <-0.05
analyze(counts, label, datasetname, methodname, threshold)

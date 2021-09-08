# Interpretability methods for differential gene analysis of scRNA-seqclustering models

This repository contains the pytorch implementation of the paper "Interpretability methods for differential gene analysis of scRNA-seqclustering models", by Madalina Ciortan.

Single-cell RNA sequencing (scRNA-seq) produces transcriptomic profilingfor individual cells.  Due to the lack of cell-class annotations, scRNA-seq is routinelyanalyzed with unsupervised clustering methods.  Because these methods are typicallylimited to producing clustering predictions, numerous model agnostic differential ex-pression (DE) libraries were proposed to identify the genes expressed differently in thedetected clusters, as needed in the downstream analysis. In parallel, the advancements inneural networks (NN) brought several model-specific interpretability methods to iden-tify salient features based on gradients, eliminating the need for external models.

We propose a comprehensive study to compare the performance of dedicated DEmethods, with that of interpretability methods typically used in machine learning, bothmodel  agnostic  (i.e.   SHAP,  permutation  importance)  and  model-specific  (i.e.   NNgradient-based methods). The DE analysis is performed on the results of 3 state-of-the-art clustering methods based on NNs. 

Our results on 36 simulated datasets indicate thatall analyzed DE methods have limited agreement between them and with ground-truthgenes.  The gradients method outperforms the traditional DE methods, which encour-ages  the  development  of  NN-based  clustering  methods  to  provide  an  out-of-the-boxDE capability.  Employing DE methods on the input data preprocessed by clusteringmethod outperforms the traditional approach of using the original count data, albeit stillperforming worse than gradient-based methods.

# Overview of the repository
- **notebooks** folder contains all jupyter notebooks to run the project, as detailed below.
- **others** folder contains the code to reproduce all experiments with scanpy, sczi, scDeepCluster
- **R** folder contains the scrips to generate the simulated data in folder R/simulated_data (both balanced and imbalanced)
- **outoput** contains model dumps and the results of running all experiments, needed to reproduce the plots
- **docker** contains the Dockerfile to create the image used to run all python experiments
- interpret_utils.py contains all funcitonalities to perform DE analysis in python
- train.py contains the main functionalities for training and evaluating the contrastive-sc model results
- model.py contains the contrastive-sc network definition
- st_loss.py contains the implementation of the loss functions for contrastive-sc
- utils.py contains various utility functions

### Overview of notebooks
- **Plots.ipynb** represents the main entry point, reproducing all the figures in the paper.
- **Contrastivesc, Groundtruth, scziDesk, scDeepCluster** contain the code to reproduce all pythond DE results

## Environment Setup
We have employed a docker container to facilitate reproducing the paper results.

### Python environment
It can be launched by running the following:

```
cd docker  
docker build -t de .
```

The image has been created for GPU usage. In order to run it on CPU, in the Dockerfile, the line "pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime" should be replaced with a CPU version.

The command above created a docker container tagged as **de** . Assuming the project has been cloned locally in a parent folder named notebooks, the image can be launched locally with:

```
docker run -it --runtime=nvidia -v ~/notebooks:/workspace/notebooks -p 8888:8888 de
```
This starts up a jupyter notebook server, which can be accessed at http://localhost:8888/tree/notebooks

### R environment

We followed the instructions on this [tutorial](http://bioinformatics.sph.harvard.edu/knowledgebase/scrnaseq/rstudio_sc_docker.html) in order to create an R docker container which comes with most single-cell related libraries already installed.
In order to launch it on port 8787, execute the following:

```
docker run -d -p 8787:8787 -e USER='rstudio' -e PASSWORD='rstudioSC' -e ROOT=TRUE -v ~/notebooks/deep_clustering:/home/rstudio/projects vbarrerab/rstudio_singlecell

```



## Data
The simulated datasets can be downloaded from this [Google Drive link](https://drive.google.com/file/d/19CSAyNgZKrX7WKoM2UP0nVNS6RHlW2I2/view?usp=sharing) (~400MB). Alternatively, it can be generated by running R/all_balanced.r or R/all_imbalanced.R.  



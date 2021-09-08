import itertools
import os
import pickle
import time
from itertools import permutations, product

import numpy as np
import pandas as pd
import scanpy.api as sc
import shap
import torch
import xgboost
from eli5.permutation_importance import get_score_importances
from scipy.stats import hypergeom
from sklearn.metrics import accuracy_score, adjusted_rand_score


def de_analysis(inputs, prefixes, data_mat, idx, method, dataset, category, clusters, nb_features = 500, run = 0, pval_cutoff = None):
    """Perform DE analysis with all python methods

    Args:
        inputs ([type]): [description]
        prefixes ([type]): [description]
        data_mat ([type]): [description]
        idx ([type]): [description]
        method ([type]): [description]
        dataset ([type]): [description]
        category ([type]): [description]
        clusters ([type]): [description]
        nb_features (int, optional): [description]. Defaults to 500.
        run (int, optional): [description]. Defaults to 0.
        pval_cutoff ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    y = np.array(data_mat['Y'])
    ari = adjusted_rand_score(y, clusters)
    print(f"ARI before: {ari}")
    de = np.array(data_mat["geneinfo"])
    de = np.array([list(x) for x in de])
    de = de[:, 4:]
    de_values = np.abs(np.log(de))
    de = (np.abs(np.log(de))>np.log(1.2)).astype(int)
    imp_f = np.where(de.sum(axis = 1)!=0)[0]

    all_results = {"features": {}, "meta": {}, "scores": {}, "features_complete": {}, "time": {}} 
    imp = pd.DataFrame(data= de_values)
    sel = np.zeros(len(imp))
    sel[idx] = 1
    imp["sel"] = sel
    all_results["features_complete"]["truth_complete"] = []
    for jj in np.sort(np.unique(y)):
        scores = np.argsort(imp[jj].values)[::-1]
        remove_0 = np.where(imp[jj].values==0)[0]
        all_results["features_complete"]["truth_complete"].append(
            np.array([el for el in scores if el not in remove_0]).astype(str))
    imp = imp[imp["sel"] ==1]
    imp = imp.drop("sel", axis =1).reset_index(drop = True)
    all_results["features"]["truth"] = []
    for jj in np.sort(np.unique(y)):
        scores = np.argsort(imp[jj].values)[::-1]
        remove_0 = np.where(imp[jj].values==0)[0]
        all_results["features"]["truth"].append(
            np.array([el for el in scores if el not in remove_0]).astype(str))
    
    imp = imp.values
    imp = (imp>0).astype(int)
    max_cluster_features = imp.sum(axis = 0)
    nb_features  = max(max_cluster_features)
    print(f"Max nb_features {nb_features}")
    selected = pd.DataFrame()
    selected["idx"] =idx
    folder = f"../output/interpretability/{category}/{method}"
    os.makedirs(folder, exist_ok=True)
    selected.to_csv(f"../output/interpretability/{category}/{method}/{dataset}_selected.csv")
    

    write_to = f"{folder}/{dataset}"
    clusters, cluster_map, _ = source_to_target_labels(clusters, y)
    ari = adjusted_rand_score(y, clusters)
    print(f"ARI after: {ari}")
    predDf = pd.DataFrame()
    predDf[method] = clusters
    predDf["groundtruth"] = y
    predDf.to_csv(f"../output/interpretability/{category}/{method}/{dataset}.csv")
    
    for i, prefix in enumerate(prefixes):
        all_results = create_features(all_results,
                            prefix,
                            write_to,
                            inputs[i],
                            clusters,
                            inputs[i].shape[1],
                            nb_features=nb_features,
                            pval_cutoff=pval_cutoff,
                            extra = { "ari": ari})
    
    all_results["meta"]["cluster_map"] = cluster_map
    all_results["meta"]["clusters"] = clusters
    all_results["meta"]["nb_features"] = max_cluster_features
    all_results["meta"]["selected_genes"] = idx
    with open(f"{write_to}_all.pkl", 'wb') as f:
        pickle.dump(all_results, f, pickle.HIGHEST_PROTOCOL)
    return all_results

def addR(results, method, nb_features, category, datasetname):
    """
    Adds the output of methods in R (e.g. EdgeR, DEsingle)
    to the python result

    Args:
        results ([type]): [description]
        method ([type]): [description]
        nb_features ([type]): [description]
        category ([type]): [description]
        datasetname ([type]): [description]
    """
    if "time" not in results.keys():
        results["time"] = {}
    i_method = "desingle"
    for i_method in ["desingle", "desinglefull"]:
        i_method_label = f"full_desingle" if i_method.endswith("full") else i_method
        column = "pvalue.adj.FDR"
        results["features"][i_method_label] = []
        results["time"][i_method_label] = 0
        s = pd.DataFrame()
        for i in np.sort(np.unique(results["meta"]["y"]).astype(int)):
            if os.path.isfile(f"../R/interpretability/{category}/{method}/{datasetname}/{i_method}_{i}.csv"):
                df = pd.read_csv(
                    f"../R/interpretability/{category}/{method}/{datasetname}/{i_method}_{i}.csv",
                    index_col=0)
                df["cluster"] = i
                df["y"] = df[column]
                df["x"] = df.index.astype(str)
                df = df.sort_values(by = "y").reset_index().iloc[:nb_features]
                if "time" in df.columns and results["time"][i_method_label] is not None:
                    results["time"][i_method_label] += df["time"].astype(float).unique()[0]
                else:
                    results["time"][i_method_label] = None
                s = s.append(df, ignore_index = True)
                if results["features"][i_method_label] is not None:
                    results["features"][i_method_label].append(df["x"].values)
            else:
                results["features"][i_method_label] = None
                results["time"][i_method_label] = None

    i_method = "edger"
    for i_method in ["edger", "edgerfull"]:
        i_method_label = f"full_edger" if i_method.endswith("full") else i_method
        column = "PValue"
        results["features"][i_method_label] = []
        results["time"][i_method_label] = 0
        s = pd.DataFrame()
        for i in np.sort(np.unique(results["meta"]["y"]).astype(int)):
            df = pd.read_csv(
                f"../R/interpretability/{category}/{method}/{datasetname}/{i_method}_{i}.csv",
                index_col=0)
            df["cluster"] = i
            df["y"] = df[column]
            df["x"] = df.index.astype(str)
            df = df.sort_values(by = "y").reset_index().iloc[:nb_features]
            results["time"][i_method_label] += df["time"].astype(float).unique()[0]
            s = s.append(df, ignore_index = True)
            results["features"][i_method_label].append(df["x"].values)
        s["rank"] = s.groupby("cluster")["y"].rank("dense", ascending=True)
        results["scores"][i_method_label] = s
        
    return results

def source_to_target_labels(source_labels, target_labels):
    """
    Rename prediction labels (clustered output) to best match true labels
    """

    source_labels, target_labels = np.array(source_labels), np.array(target_labels)
    assert source_labels.ndim == 1 == target_labels.ndim
    unique_source_labels = np.unique(source_labels)
    accuracy = -1
    if len(np.unique(source_labels)) != len(np.unique(target_labels)):
        perms = product(np.unique(target_labels), repeat = len(unique_source_labels))
    else:
        perms = np.array(list(permutations(unique_source_labels)))
    best_perm = None
    remapped_labels = target_labels
    for perm in perms:
        cmap = dict(zip(unique_source_labels, perm))
        f = lambda x: cmap[x]
        
        flipped_labels = np.vectorize(f)(source_labels)
        testAcc = accuracy_score(target_labels, flipped_labels)
        if testAcc > accuracy:
            accuracy = testAcc
            remapped_labels = flipped_labels
            best_perm = perm

    source_to_target = dict(zip(unique_source_labels, best_perm))
    return remapped_labels, source_to_target, accuracy

def create_features(all_features,
                    prefix,
                    name,
                    X,
                    clusters,
                    total_size,
                    nb_features=20,
                    pval_cutoff=0.01,
                    extra={}):
    """Compute the DE genes with all pythin methods (statistical tests,
    shap, feature permutation)

    Args:
        all_features ([type]): [description]
        prefix ([type]): [description]
        name ([type]): [description]
        X ([type]): [description]
        clusters ([type]): [description]
        total_size ([type]): [description]
        nb_features (int, optional): [description]. Defaults to 20.
        pval_cutoff (float, optional): [description]. Defaults to 0.01.
        extra (dict, optional): [description]. Defaults to {}.
    """

    start = time.time()
    df1, features1 = get_feature_ranks(X,
                                       clusters,
                                       method='t-test',
                                       pval_cutoff=pval_cutoff,
                                       nb_features=nb_features)
    end = time.time()
    all_features["time"][f"{prefix}t-test"] = end - start
    all_features["features"][f"{prefix}t-test"] = features1
    all_features["scores"][f"{prefix}t-test"] = df1

    start = time.time()
    df2, features2 = get_feature_ranks(X,
                                       clusters,
                                       method='t-test_overestim_var',
                                       pval_cutoff=pval_cutoff,
                                       nb_features=nb_features)
    end = time.time()
    all_features["time"][f"{prefix}t-test_overestim_var"] = end - start
    all_features["features"][f"{prefix}t-test_overestim_var"] = features2
    all_features["scores"][f"{prefix}t-test_overestim_var"] = df2

    start = time.time()
    df3, features3 = get_feature_ranks(X,
                                       clusters,
                                       method='wilcoxon',
                                       pval_cutoff=pval_cutoff,
                                       nb_features=nb_features)
    end = time.time()
    all_features["time"][f"{prefix}wilcoxon"] = end - start
    all_features["features"][f"{prefix}wilcoxon"] = features3
    all_features["scores"][f"{prefix}wilcoxon"] = df3


    start = time.time()
    df, features = xgboost_feature_permutations(X,
                                                clusters,
                                                nb_features=nb_features)
    end = time.time()
    all_features["time"][f"{prefix}xgboost_permutations"] = end - start
    all_features["features"][f"{prefix}xgboost_permutations"] = features
    all_features["scores"][f"{prefix}xgboost_permutations"] = df

    start = time.time()
    df, features = xgboost_shap(X, clusters, nb_features=nb_features)
    end = time.time()
    all_features["time"][f"{prefix}xgboost_shap"] = end - start
    all_features["features"][f"{prefix}xgboost_shap"] = features
    all_features["scores"][f"{prefix}xgboost_shap"] = df
    
    all_features["meta"][f"{prefix}feature_size"] = X.shape[1]
    all_features["meta"][f"{prefix}extra"] = extra
    all_features["meta"]["y"] = clusters

    with open(f"{name}_all.pkl", 'wb') as f:
        pickle.dump(all_features, f, pickle.HIGHEST_PROTOCOL)
    return all_features
    
def xgboost_feature_permutations(X, clusters, nb_features = 20):
    """ Compute the feature permutations interpretability method.

    Args:
        X ([type]): [description]
        clusters ([type]): [description]
        nb_features (int, optional): [description]. Defaults to 20.

    Returns:
        [type]: [description]
    """
    model = xgboost.XGBClassifier().fit(pd.DataFrame(X), clusters)
    def score(X, y):
        y_pred = model.predict(X)
        return (y_pred ==y).astype(int)
    
    base_score, score_decreases = get_score_importances(score, X, clusters)
    feature_importances = np.mean(score_decreases, axis=0)

    features  = []
    df_tot = None
    for c in np.sort(np.unique(clusters)):
        idx = np.where(clusters == c)[0]
        df = pd.DataFrame()
        df["y"] = list(np.sort(feature_importances.T[idx].mean(axis = 0)
                                  )[::-1][:nb_features])
        df["x"] = list(np.argsort(feature_importances.T[idx].mean(axis = 0)
                                  )[::-1][:nb_features].astype(str))
        df["cluster"] = c
        features.append(df["x"].values)
        if df_tot is None:
            df_tot =df
        else:
            df_tot= df_tot.append(df, ignore_index = True)

    df_tot["rank"] = df_tot.groupby("cluster")["y"].rank("dense", ascending=False)
    return df_tot, features

def xgboost_shap(X, clusters, nb_features = 20):
    """Implements SHAP
    https://slundberg.github.io/shap/notebooks/tree_explainer/Census%20income%20classification%20with%20LightGBM.html

    Args:
        X ([type]): [description]
        clusters ([type]): [description]
        nb_features (int, optional): [description]. Defaults to 20.
    """

    model = xgboost.XGBClassifier().fit(X, clusters)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_values = np.array(shap_values)
    if len(shap_values.shape) ==3:
        feature_importances = shap_values.mean(axis = 0)
    else:
        feature_importances = shap_values
    print(f"feature_importances {feature_importances.shape}")
    features  = []
    df_tot = pd.DataFrame()
    for c in np.sort(np.unique(clusters)):
        idx = np.where(clusters == c)[0]
        df = pd.DataFrame()
        if len(feature_importances.shape)>1:
            df["y"] = list(np.sort(feature_importances[idx].mean(axis = 0)
                                      )[::-1][:nb_features])
            df["x"] = list(np.argsort(feature_importances[idx].mean(axis = 0)
                                      )[::-1][:nb_features].astype(str))
        df["cluster"] = c
        features.append(df["x"].values)

        df_tot= df_tot.append(df, ignore_index = True)

    df_tot["rank"] = df_tot.groupby("cluster")["y"].rank("dense", ascending=False)
    return df_tot, features

def analyze(dict_features, total_nb_features, nb_features_truth = 0):
    """ Compute accuracy of various DE methods.

    Args:
        dict_features ([type]): [description]
        total_nb_features ([type]): [description]
        nb_features_truth (int, optional): [description]. Defaults to 0.
    """
    features_orig = list(dict_features.values())
    names_orig = list(dict_features.keys())
    # remove None elements which correspond to entries half processed
    features, names = [], []
    for i in range(len(features_orig)):
        if features_orig[i] is not None:
            features.append(features_orig[i])
            names.append(names_orig[i])
            
    results = {"iou": [], "precision": [], "overlap": [], "counts": []}
    ratios = pd.DataFrame()
    for c in range(len(features[0])): # for each cluster
        if nb_features_truth ==0:
            nb_features_truth = len(dict_features["truth"][c])

        cluster_data = [x[c][:nb_features_truth] for x in features]
#         m = distance_matrix(
#             cluster_data,
#             lambda u, v: compute_iou(u, v),
#             symmetric=True)
#         m = pd.DataFrame(m)
#         m.columns = names
#         m["names"] = names
#         m = m.set_index("names")
#         results["iou"].append(m)

#         m = distance_matrix(
#             cluster_data,
#             lambda u, v: compute_precision(u, v),
#             symmetric=False)
#         m = pd.DataFrame(m)
#         m.columns = names
#         m["names"] = names
#         m = m.set_index("names")
#         results["precision"].append(m)

#         m = distance_matrix(
#             cluster_data, lambda u, v: compute_hypergeom(
#                 u, v, nb_features_truth))
#         m = pd.DataFrame(m)
#         m.columns = names
#         m["names"] = names
#         m = m.set_index("names")
#         results["overlap"].append(m)
        
        m = distance_matrix(
            cluster_data, lambda u, v: compute_counts(
                u, v), symmetric=False)
        m = pd.DataFrame(m)
        m.columns = names
        m["names"] = names
        m = m.set_index("names")
        results["counts"].append(m)

        ratio = m["truth"]/m["truth"].max()# scale values between 0 and 1
        ratio = ratio.reset_index()
#         ratio["cluster"] = c
        ratio = ratio[ratio["names"] != "truth"]
        ratios = ratios.append(ratio.set_index("names").T, ignore_index = True)
    results["acc"] = ratios
    return results

def distance_matrix(ls, f, symmetric = True):
    A = np.zeros((len(ls), len(ls)))
    for i,j in list(itertools.combinations(np.arange(len(ls)),2)):
        A[i, j] = f(ls[i], ls[j])
        if symmetric:
            A[j, i] = A[i, j]
        else:
            A[j, i] = f(ls[j], ls[i])
    for i in np.arange(len(ls)):
        A[i, i] = f(ls[i], ls[i])

    return A

def overlap(subspace1_size, subspace2_size, overlap_size, dataset_size):
    return hypergeom.sf(overlap_size-1, dataset_size, subspace1_size, subspace2_size)

def compute_hypergeom(arr1, arr2, total_size):
    intersect = np.intersect1d(arr1, arr2)
#     print(len(arr1), len(arr2), len(intersect), total_size)
    score = overlap(len(arr1), len(arr2), len(intersect), total_size)
    return score
    
def compute_precision(arr1, arr2):
    intersect = np.intersect1d(arr1, arr2)
    precision1 = len(intersect)/len(arr1)
    return precision1


def compute_counts(arr1, arr2):
    intersect = np.intersect1d(arr1, arr2)
    return len(intersect)

def compute_iou(arr1, arr2):
    intersect = np.intersect1d(arr1, arr2)
    union = np.unique(np.concatenate([arr1, arr2]))
    iou = len(intersect)/len(union)
    return iou



def get_feature_ranks(X, clusters, method = 't-test', pval_cutoff = None, nb_features = 50):
    """
    method = ['t-test', 't-test_overestim_var', 'wilcoxon', 'logreg']
    """
    
    adata = sc.AnnData(X)
    adata.obs['kmeans_pred'] = pd.Series(clusters).astype('category').values
    sc.tl.rank_genes_groups(adata,
                            'kmeans_pred',
                            method=method,
                            key_added=method,
                            n_genes = nb_features)
    
    properties = ['scores', 'names']

    for p in ['pvals_adj', 'pvals',  'logfoldchanges']:
        if p in list(adata.uns[method].keys()):
            properties.append(p)
            
    column = "pvals_adj" if "pvals_adj" in properties else "scores"

    df = None
    for c in np.unique(clusters):
        d = pd.DataFrame()
        for k in properties: 
            d[k] = adata.uns[method][k][f'{c}']

        d['cluster'] = c
        d['x'] = d['names']
        d = d.sort_values(by=column).reset_index(drop = True).iloc[:nb_features]
        if df is None:
            df = d
        else:
            df = pd.concat([df, d], ignore_index = True)
            
    
    if pval_cutoff is not None and 'pvals_adj' in df.columns:
        df = df[df["pvals_adj"]<pval_cutoff]
        
#     df = df.sort_values(by="pvals_adj").reset_index(drop = True)
#     if percentile is not None:
#         nb = df.shape[0] - df.shape[0]*percentile//100
#         df = df.sort_values(by="pvals_adj").reset_index(drop = True).iloc[:nb]
    features = []
    
    df["y"] = df[column]
    df["rank"] = df.groupby("cluster")["y"].rank("dense", ascending=True)
    for cluster, group in df.groupby("cluster"):

        features.append(group.sort_values(by=column)["names"].values[:nb_features])
#     features = [list(x[0]) for x in df.groupby("cluster").agg({"names": list}).values]
    return df, features

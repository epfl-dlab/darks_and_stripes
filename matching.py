import pandas as pd
import numpy as np
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import scipy.stats as spst
import scipy as sp
import sys
import networkx as nx
from networkx.algorithms import bipartite
                  
# motivates by Just Noticable Difference, ca.1
bmi_bound = 1.0

def build_est_matching_pairs(images):      
    matches = {}
    images = images.copy()
    # matching is normally done on the true labels, but here we want to match on estimates 
    #images["bmi_true"] = images["bmi_est"].values
    
    print(images.bmi_true)
    
    for i in images.index.values:
        img = images.loc[i]
        sim_img = images.loc[(images.gender == img.gender) &\
                             (images.w_est < img.w_est+2) & (images.w_est > img.w_est-2) &\
                             (images.h_est < img.h_est+2) & (images.h_est > img.h_est-2) &\
                             (images.bmi_est < img.bmi_est+1.) & (images.bmi_est > img.bmi_est-1.)]
        sim_img = sim_img.drop(i)
        matches[i] = sim_img.index.values
    
    pairs = set()    
    for i, m in matches.items():
        for j in m:
            ior = i#.replace("_light","").replace("_dark","")
            jor = j#.replace("_light","").replace("_dark","")
            pairs.add((min(ior,jor), max(ior,jor)))
            
    return pairs

def get_matching_images(img, images, bound):
    
    sim_img = images.loc[(images.gender == img.gender) &\
                         (images.w_true < img.w_true+bound) & (images.w_true > img.w_true-bound) &\
                         (images.h_true < img.h_true+bound) & (images.h_true > img.h_true-bound) &\
                         (images.bmi_true < img.bmi_true+bmi_bound) & (images.bmi_true > img.bmi_true-bmi_bound)]
    
    return sim_img
                      
# input: two dataframes - all images and treated images to be matched 
# returns: two lists - names of sample and matched images
# matching is performed based on covariates: gender, weight, height in a greedy fashion 
def match_on_covariates(images, sample_images, bound):
    images = images.drop(sample_images.index.values, errors='ignore')
    img_matched = []
    for i, img in sample_images.iterrows():
        sim_img = get_matching_images(img, images, bound)
        #sim_img["diff"] = (sim_img.w_true - img.w_true)**2 + (sim_img.h_true - img.h_true)**2
        sim_img["diff"] = (sim_img.bmi_true - img.bmi_true)**2
        sim_img = sim_img.sort_values(by="diff")
        
        if (sim_img.shape[0] > 0):
            img_matched.append(sim_img.iloc[0].name)
            images = images.drop(sim_img.iloc[0].name)
        else:
            img_matched.append("")
    
    matched = (np.asarray(img_matched) != "")
    
    print(len(img_matched), len(np.unique(img_matched)))
    
    return np.asarray(sample_images.index.values)[matched==True], np.asarray(img_matched)[matched==True]    
    
    
def max_match_on_covariates(images, sample_images, bound):
    
    images = images.drop(sample_images.index.values, errors='ignore')
    G = nx.Graph()
    
    G.add_nodes_from(sample_images.index.values, bipartite=0)
    G.add_nodes_from(images.index.values, bipartite=1)
    
    for i, img in sample_images.iterrows():
        sim_img = get_matching_images(img, images, bound)
        
        for j in sim_img.index.values:
            G.add_edge(i, j)
        
    #match = nx.max_weight_matching(G, maxcardinality=True)
    match = nx.bipartite.maximum_matching(G, top_nodes = sample_images.index.values)
    
    sample_img, matched_img = [], []
    #print("Matched size")
    #print(match)
    
    for a, b in match.items():
        #print(a,b)
        if (b in sample_images.index.values):
            continue
        sample_img.append(a)
        matched_img.append(b)
        #print(sample_images.loc[a,"w_true"], images.loc[b,"w_true"])
    
    return np.asarray(sample_img), np.asarray(matched_img)    

        
    
# fit a logit model to predict propensity scores: P(Z_k == 1 | X_k)
def compute_propensity_score(data, cov_names, trmnt_name):
    lr = LogisticRegression()
    
    print(cov_names)
    print(trmnt_name)
    lr.fit(data[cov_names].values, data[trmnt_name].values)
    data["prop_score"] = lr.predict_proba(data[cov_names].values)[:,1]
    return data
    
def match_on_propensity_score(images, sample_images):
    images = images.drop(sample_images.index.values, errors='ignore')
    img_matched = []
    # or take all of them
    for i, img in sample_images.iterrows():    
        sim_img = images
        sim_img["diff"] = (sim_img.prop_score - img.prop_score)**2
        sim_img = sim_img.sort_values(by="diff")
    
        img_matched.append(sim_img.iloc[0].name)
        images = images.drop(sim_img.iloc[0].name)
    
    return np.asarray(sample_images.index.values), np.asarray(img_matched)
    
        
# look at groups composed of m pairs and see whether the treated samples stand out
# lets us detect rear but stong treatment effects
def m_group_effects(m, r_trt, r_ctr):
    pos_effect = r_trt > r_ctr
    ranks = np.argsort(np.absolute(r_trt - r_ctr))
    
    # compute the statistics t for the data at hand
    # compute E[t], Var[t] under H_0 of no effect
    t, Et, Vt = 0, 0, 0 
    for i in range(m, r_trt.shape[0]+1):
        cnt = sp.misc.comb(i-1, m-1)
        Et += 0.5*cnt
        Vt += 0.25*(cnt**2)
        if (pos_effect[ranks[i-1]]):
            t += cnt
            
    # compute a 95% upper bound on t
    t95 = Et + np.sqrt(Vt)*sp.stats.norm.ppf(0.95)
    cnt_all = sp.misc.comb(r_trt.shape[0], m)
    
    p = sp.stats.norm.cdf((t-Et)/np.sqrt(Vt))
    
    return 1-p, t/cnt_all, t95/cnt_all
        
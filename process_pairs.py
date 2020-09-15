import pandas as pd
import numpy as np
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
from sklearn.linear_model import LogisticRegression
import scipy.stats as spst
import sys
import re
import collections

from routines import *
from matching import *

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

def clear_imgname(imname):
    return imname.replace("_dark","").replace("_light","")

def compute_votes(im1, im2, est):
    est = est.loc[est.pairname == im1.name+im2.name]
    v1 = est.loc[est.num == im1.num].shape[0]
    v2 = est.loc[est.num == im1.num].shape[0]
    return v1, v2

def plot_image_pair(im1, im2, est):    
    v1, v2 = compute_votes(im1, im2, est)
    plt.subplot(1,2,1)
    plt.imshow(mpimg.imread('_all_/'+im1.name))
    plt.title("True: %.2lf cm, %.2lf kg\nVotes: %d" % (im1.h_true, im1.w_true, v1), fontsize=5)
    plt.subplot(1,2,2)
    plt.imshow(mpimg.imread('_all_/'+im2.name))
    plt.title("True: %.2lf cm, %.2lf kg\nVotes: %d" % (im2.h_true, im2.w_true, v2), fontsize=5)
    plt.axis('off')

def get_group(f):
    '''
    if ("_light" in f or "_dark" in f):
        return "n"
    else:
        return "s"
    '''
    if ("_light" in f):
        return "l"
    elif ("_dark" in f):
        return "d"
    else:
        return "s"
    
def assign_color_groups(est, images):
    est["pairname"] = [x.file1+x.file2 for _, x in est.iterrows()] 
    est["group"] = [get_group(x.file1) + get_group(x.file2) for _, x in est.iterrows()]
    print(set(est.num.values) - set(images.num.values))
    est["vote"] = [images.loc[images.num == x.num].index.values[0] for _, x in est.iterrows()]
    #est["votegroup"] = [x.vote.split("_")[-2][0] for _, x in est.iterrows()]
    
    print(est.groupby("group").apply(lambda x: x.shape[0]))
    
    return est
    
def compute_stats(est, imgpairs):
    
    imgpairs["votes1"], imgpairs["votes2"] = 0.5, 0.5
    imgpairs["group"] = ""
    
    # CIs based on the per-images vote distribution.
    for ip, p in imgpairs.iterrows():
        cest = est.loc[est.pairname == p.img1+p.img2]
        if (cest.shape[0] > 0):
            imgpairs.loc[ip,"group"] = cest.group.values[0]
            imgpairs.loc[ip,"votes1"] = cest.loc[cest.vote == p.img1].shape[0]/cest.shape[0]
            imgpairs.loc[ip,"votes2"] = cest.loc[cest.vote == p.img2].shape[0]/cest.shape[0]
        
    #estcnt = est.groupby("pairname").apply(lambda x: )
    
    gdv = {}
    for g in np.unique(imgpairs.group.values):
        cres = imgpairs.loc[imgpairs.group == g]
        gdv[g] = bs.bootstrap(cres.votes1.values, stat_func=bs_stats.mean, alpha=0.05, num_iterations=10000)

        
    g = np.sort(np.unique(imgpairs.group.values))
    
    if (g[0] == ""):
        g = g[1:]
    
    print("Count of images per group:", imgpairs.groupby("group").apply(lambda x: x.shape[0]))
    

    # CIs based on separate resampling of votes for each image.
    means = collections.defaultdict(list)
    allgroups = np.unique(imgpairs.group.values)
    for count in range(10000):
        if (count % 100 == 0):
            print(count)
        cmeans = collections.defaultdict(list)
        for ip, p in imgpairs.iterrows():
            cest = est.loc[est.pairname == p.img1+p.img2]
            if (cest.shape[0] == 0):
                continue
            cmeans[cest.group.values[0]].append(np.mean(np.random.choice(cest.vote == p.img1, 40, replace=True)))
        for group in allgroups:
            means[group].append(np.mean(cmeans[group]))
            
    for group in allgroups:
        pCI = np.percentile(means[group], [0, 95])        
        print("%s: %.3lf (%.3lf, %.3lf)" % (group, np.mean(imgpairs.loc[imgpairs.group == group, "votes1"]), pCI[0], pCI[1]))

    sys.exit(1)
    return imgpairs, g, gdv
    
def compute_vote_change(img4, ig):
    img4 = img4.copy()
    img4 = img4.set_index('group')
    # see what happens to the votes when any person changes into light
    if (ig == "dd"):
        return (img4.loc["ld","vote1"] - img4.loc["dd","vote1"]) +\
               (img4.loc["dl","vote2"] - img4.loc["dd","vote2"])
    
def plot_pair_results(imgpairs):

    plt.subplot(2,4,1)
    plt.imgshpw(im)
    imgpairs["pairname"] = [clear_imgname(x.img1)+clear_imgname(x.img2) for _, x in imgpairs.iterrows()]
    print(imgpairs[["img1","img2"]])
    
    #vupd = imgpairs.groupby("pairname").apply(compute_vote_change(x, ig))
    
    #for ip, p in imgpairs.iterrows():
        
        
    
if __name__ == "__main__":
    
    folder = sys.argv[1]
    
    est = pd.read_csv(folder+os.sep+'est.csv', index_col=0)
    workers = pd.read_csv(folder+os.sep+'workers.csv', index_col=0)
    images = pd.read_csv(folder+os.sep+'images.csv', index_col=0)
    imgpairs = pd.read_csv(folder+os.sep+'matched_pairs.csv', index_col=0)
    
    # plot settings
    sns.set(style="whitegrid", palette="bright", color_codes=True)
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']
    rcParams['figure.autolayout'] = True
    
    # set the random seed to make results reproducable
    np.random.seed(239)
    
    est = assign_color_groups(est, images)   
    imgpairs = compute_stats(est, imgpairs)
    
    plot_change(imgpairs, "dd")
    

        
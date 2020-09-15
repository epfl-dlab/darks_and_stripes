import pandas as pd
import numpy as np
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as spst
import scipy as sp
import sys
from matplotlib import rcParams


def simulate_random_guesses(ws_v, N = 1000):    
    estnum = len(ws_v)
    ws_mean = np.zeros((N,estnum))
            
    for i in range(N):
        ws_sample = np.random.choice(ws_v, estnum, replace=False)
        ws_mean[i,:] = np.asarray([np.mean(ws_sample[:(i+1)]) for i in range(0,ws_sample.shape[0])])
    
    for i in range(estnum):
        ws_mean[:,i] = np.sort(ws_mean[:,i])
    
    return ws_mean
    

def plot_time_evolution(est, images):
    # random subset of images
    im_idx = []
    cnt_v_min = 0
    cnt_v_max = 60
    while (cnt_v_min < 70 or cnt_v_max > 150):
        im_ids = np.random.choice(images.shape[0], 2)
        cnt_est = est.loc[est.filename.isin(images.iloc[im_ids].index.values)].groupby("filename").apply(lambda x: x.shape[0])
        print(cnt_est)
        cnt_v_min = np.min(cnt_est.values)
        cnt_v_max = np.max(cnt_est.values)
        
    
    print(cnt_v_min, cnt_v_max)
    
    for i in im_ids:
        image = images.iloc[i]
        ws_v = est.loc[(est.filename==image.name),'w'].values
        estnum = len(ws_v)
        print(ws_v)
        ws_mean = simulate_random_guesses(ws_v)
        
         
        plt.plot([1, estnum], [np.mean(ws_v), np.mean(ws_v)], label="Mean of all 75 estimates") 
        plt.plot(range(1, estnum+1), ws_mean[25], color='black', linestyle='dashed', label="95% CI for n estimates")
        plt.plot(range(1, estnum+1), ws_mean[975], color='black', linestyle='dashed')
        
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel("Mean weight estimate [kg]", fontsize=18)
        plt.xlabel("Number n of estimates", fontsize=18)
        plt.savefig('Plots/plot_convergence_'+str(i)+'.pdf')
        plt.show()
    
       
if __name__ == "__main__":
    # plot settings
    sns.set(style="whitegrid", palette="bright", color_codes=True)
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']
    rcParams['figure.autolayout'] = True
    
    
    folder = 'large_set_results'
    est, images, workers = pd.read_csv(folder + "/est.csv", index_col = 0),\
                           pd.read_csv(folder + "/images_final.csv", index_col = "filename"),\
                           pd.read_csv(folder + "/workers.csv", index_col = "workerId")
                         
    images = images.loc[images.gender.isin(["female","male"])]
    images = images.loc[images.w_true > 30]
    images = images.loc[images.h_true > 100]  
    est = est.loc[est.filename.isin(images.index.values)]
    
    plot_time_evolution(est, images)
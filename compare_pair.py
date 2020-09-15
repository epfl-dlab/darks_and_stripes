import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import numpy as np

def read_clean_data():
    images0, est0, workers0 = pd.read_csv('clean_set_results/res_clothes_dark/images.csv', index_col = 0),\
                              pd.read_csv('clean_set_results/res_clothes_dark/est.csv', index_col = 0),\
                              pd.read_csv('clean_set_results/res_clothes_dark/workers.csv', index_col = 0)
    
    images1, est1, workers1 = pd.read_csv('clean_set_results/res_clothes_light/images.csv', index_col = 0),\
                              pd.read_csv('clean_set_results/res_clothes_light/est.csv', index_col = 0),\
                              pd.read_csv('clean_set_results/res_clothes_light/workers.csv', index_col = 0)
    
    images2, est2, workers2 = pd.read_csv('clean_set_results/res_clothes_stripes/images.csv', index_col = 0),\
                              pd.read_csv('clean_set_results/res_clothes_stripes/est.csv', index_col = 0),\
                              pd.read_csv('clean_set_results/res_clothes_stripes/workers.csv', index_col = 0)
    
    return pd.concat([images0, images1, images2], axis=0), pd.concat([est0, est1, est2], axis=0), pd.concat([workers0, workers1, workers2], axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, required=True)
    parser.add_argument('--file2', type=str, required=True)
    
    args = parser.parse_args()
    
    images, est, workers = read_clean_data()
    
    images = images.loc[[args.file1, args.file2]]
    #est = est.loc[est.filename.isin([args.file1, args.file2])]
    workers = workers.loc[workers.country=="United States"]
    est = est.loc[est.workerId.isin(workers.index.values)]
    
    workers = workers.loc[est.loc[est.filename.isin([args.file1, args.file2]),"workerId"].values]
    workers1, workers2 = workers.loc[est.loc[est.filename == args.file1,"workerId"].values],\
                         workers.loc[est.loc[est.filename == args.file2,"workerId"].values]
    
    est = est.loc[est.workerId.isin(workers.index.values)]
    
    w_mean_guess = est.groupby('workerId').apply(lambda x: np.mean(x.w))
    w1, w2 = w_mean_guess[workers1.index.values], w_mean_guess[workers2.index.values]
    
    print(np.sort(w1.values))
    print(np.sort(w2.values))
    
    #w1_mean_guess, w2_mean_guess = pd.sort_values(by='')
    
    plt.subplot(1,2,1)
    
    w1, w2 = est.loc[est.filename == args.file1,"w"].values, est.loc[est.filename == args.file2,"w"].values
    w1, w2 = np.sort(w1), np.sort(w2)
    
    print("10 largest:")
    print(w1[-10:])
    print(w2[-10:])
    print(np.mean(w1), np.mean(w2))
    
    sns.distplot(est.loc[est.filename == args.file1,"w"], label=args.file1)
    sns.distplot(est.loc[est.filename == args.file2,"w"], label=args.file2)
    plt.legend()
    
    plt.subplot(1,2,2)
    #sns.distplot(est.loc[est.filename == args.file1,"h"], label=args.file1)
    #sns.distplot(est.loc[est.filename == args.file2,"h"], label=args.file2)
    sns.distplot(workers1.w_own, label=args.file1)
    sns.distplot(workers2.w_own, label=args.file2)
    plt.legend()
    print(np.mean(workers1.w_own), np.mean(workers2.w_own))
    plt.show()
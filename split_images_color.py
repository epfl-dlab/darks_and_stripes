import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil
import matplotlib.pyplot as plt
  
def split_color_groups(images):
    images["color"] = images[["r","g","b"]].mean(axis=1)
    
    images = images.loc[images.r >= 0]
    
    images = images.sort_values(by="color")
    dark_bdr = images.color[300]#round(0.25*images.shape[0])]
    light_bdr = images.color[images.shape[0]-300]#round(0.75*images.shape[0])]
    
    images["color_type"] = "none"
    images.loc[images.color <= dark_bdr,"color_type"] = "dark"
    images.loc[images.color >= light_bdr,"color_type"] = "light"
    
    for (pn, q) in enumerate([0,0.33,0.66,1.]):
        i = min(round(q*images.shape[0]), images.shape[0]-1)
        plt.subplot(1,4,pn+1)
        imname = images.index.values[i]
        imname = imname.split('.')[0] + "_detected.jpg"
        X = plt.imread("_all_/"+imname)
        plt.imshow(X, aspect=0.0022*X.shape[1])
        plt.axis('off')
        plt.title("bv = %.2lf" % images.iloc[i].color, fontsize=9)
        
    #plt.tight_layout()
    plt.savefig("color_quantiles.pdf")
    return images
    
if __name__ == "__main__":
    
    images = pd.read_csv("color_detection/mturk_res/images_proc_cd.csv", index_col = "filename")
    images = images.loc[images.naked == 0]
    
    images = split_color_groups(images.copy())
    '''
    print(images)
    for color in ["light", "dark"]:
        for i, im in images.loc[images.color_type == color].iterrows():
            shutil.copyfile("_all_/"+i, "_"+color+"_/"+i)
        
    images.to_csv("mturk_results/images_proc_colorgroups.csv")
    
    im_err = pd.read_csv("mturk_results/images_proc_errors.csv", index_col = "filename")
    im_err["color_type"] = images.color_type
    #images = pd.concat(im_err, images[["r","g","b","color_type"]], axis=1, sort=True, join='outer')
    
    print(im_err)
    
    im_err.to_csv("mturk_results/images_final.csv", float_format="%.2lf")
    '''
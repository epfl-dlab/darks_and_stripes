import pandas as pd
import numpy as np
from pathlib import Path
import os

def read_data(folder):    
    return pd.read_csv(folder + "/est.csv", index_col = 0),\
            pd.read_csv(folder + "/images.csv", index_col = "filename"),\
            pd.read_csv(folder + "/workers.csv", index_col = "workerId")
    
def split_var_folder(images, var_name, folder):    
    img_folder = []
    for i in images.index.values:
        img_file = Path(folder + os.sep + i)
        if img_file.is_file():
            img_folder.append(i)
     
    print(img_folder)
    images[var_name] = 0       
    images.loc[img_folder,var_name] = 1
    
    return images
    
    
if __name__ == "__main__":
    
    est, images, workers = read_data("mturk_results")
    
    images = split_var_folder(images, "stripes", "_stripes_manual_")
    images = split_var_folder(images, "naked", "_naked_")
    
    images.to_csv("mturk_results/images_proc.csv", float_format="%.2f")
    
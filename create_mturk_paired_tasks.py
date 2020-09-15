import pandas as pd
import numpy as np
import math

def build_header(hitsize):
    s = "im_sample0,im_sample1"
    for i in range(hitsize):
        s += ",im_left" + str(i)
        s += ",numleft" + str(i)
        s += ",im_right" + str(i)
        s += ",numright" + str(i)
    s += "\n"
    return s

# 2xhitsize array of imgnames, 
def build_task(imgnames, imgnums):
    url_base = "http://51.15.251.127/data/"
    # sample images
    #s = url_base + "POU3xCt_light_numbered.jpg"
    #s += "," + url_base + "psb1auckrn611_light_numbered.jpg"
    s = url_base + "0GGnhSl_2_light_numbered.jpg"
    s += "," + url_base + "CEC3Nnm_1_numbered.jpg"
    for i in range(imgnames.shape[0]):
        s += "," + url_base + imgnames[i,0]
        s += "," + imgnums.loc[imgnames[i,0],"num"]
        s += "," + url_base + imgnames[i,1]
        s += "," + imgnums.loc[imgnames[i,1],"num"]
    s += "\n"
    return s 

def check(imgnames, d):
    cleaner = lambda t: t.replace("_light","").replace("_dark","")
    cleanerall = np.vectorize(cleaner)
    cn = cleanerall(imgnames)
        
    for i in range(0, cn.shape[0]-d):
        for j in range(1,d):
            if (cn[i+j,0] == cn[i,0] and cn[i+j,1] == cn[i,1]):
                print("Same within", j)
                return False
        
    return True
    

if __name__ == "__main__":
    
    imgpairs = pd.read_csv('matched_pairs.csv', index_col=0)
    imgnums = pd.read_csv("_photoshop_/images_numbered.csv", index_col=0)
    
    hitsize=10
    numhits = math.ceil((imgpairs.shape[0]/4)/hitsize)        
        
    # 4 tasks with different couples
    cnt_matches = int(imgpairs.shape[0]/4)
    idxrand = np.random.choice(4, cnt_matches)
    idxbase = np.array([4*x for x in range(cnt_matches)])
    allidx = set()

    for add in range(4):
        idxrand += add
        idxrand %= 4
        idx = idxrand + idxbase
        allidx = allidx.union(idx)
        
        # append additional tasks without creating too small distance between the same pairs
        while (True):
            print("Tossing in", round(numhits*hitsize - imgpairs.shape[0]/4), "repeating pairs")
            idxs = np.append(idx, np.random.choice(idx, round(numhits*hitsize - imgpairs.shape[0]/4), replace=False))
            np.random.shuffle(idxs)
            if (check(imgpairs.loc[idxs].values, d=10)):
                break
        print("Done!") 
               
        f = open('mturk_input_paired_height' + str(add) + ".csv", 'w')
        f.write(build_header(hitsize))  # python will convert \n to os.linesep
        for h in range(numhits): 
            cidxs = idxs[h*hitsize:(h+1)*hitsize]
            f.write(build_task(imgpairs.loc[cidxs].values, imgnums))
        f.close()
    print(len(allidx))    
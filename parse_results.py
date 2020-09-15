import csv
import numpy as np
import pandas as pd
import heapq
import sys
import itertools
import math

folderout = ""

def get_hitsize(colnames):
    hitsize = 0    
    while (("Input.im_left" + str(hitsize)) in colnames):
        hitsize += 1;
    return hitsize
    
def url_to_filename(url):
    return url.split('/')[-1]
            
def init_datasets(data, survey):
    # collect all image names and init images data frame
    hitsize = get_hitsize(data.columns)
    
    # images
    images = pd.read_csv(folderout + "/images.csv", index_col=0)

    # workers 
    workers = pd.DataFrame(index=np.unique(data.WorkerId))
    workers.index.name = 'workerId'
    workers['lbs'], workers['feet'], workers['inch'] = 0, 0, 0
    for s in survey:
        workers[s] = "none"
    workers['browser_info'] = "none"
    workers['hit_count'], workers["mean_time"], workers["cnt_invalid_survey"] = 0, 0, 0   
    
    # HITs
    hits = pd.DataFrame(index=np.unique(data.HITId))
    hits["filenames"] = [set() for _ in hits.index]
    
    # estimates
    est = pd.DataFrame(index = range(hitsize*data.shape[0]))
    numest = est.shape[0]
    est["file1"], est["file2"], est["workerId"] = "", "", ""
    est["assignmentId"], est["hitId"] = "", ""
    est["t"], est["valid"] = 0, 0
        
    return est, images, workers, hits
        
def compute_HIT_bonus(est, images, workers, hits):
    # compute statistics and bonuses for each HIT    
    workers['bonus_ids'] = [list() for x in workers.index]
    print(type(workers.bonus_ids.values[0]))    
    hitsize = len(hits.iloc[0,0])    
    tot_boni = 0
    for hitId, hit in hits.iterrows():
        w_err = est.loc[est.hitId == hitId,["workerId", "correct"]]
        w_err = w_err.groupby("workerId").apply(lambda x: np.sum(x.correct))
        # best 25% of workers receive a bonus
        k = int(round(0.15*w_err.shape[0]))
        bonus_err_vals = heapq.nsmallest(k, w_err.values)
        
        bonus_workers = w_err.isin(bonus_err_vals)
        bonus_workers = bonus_workers[bonus_workers == True].index
        
        tot_boni += len(bonus_workers)
        for worker in bonus_workers:
            assignId = est.loc[(est.workerId == worker) & (est.hitId==hitId), "assignmentId"]
            workers.loc[worker,"bonus_ids"].append(assignId.values[0])
    print("Total boni for", tot_boni, "HITs out of", est.shape[0]/hitsize)
    return hits, workers
    
def check_workers_validity(est, workers):
    workers["mean_time"] = est.groupby("workerId").apply(lambda x: np.mean(x.t))

    # leave out potential cheaters -> those who change their answers to questions more than once
    print("\n\nInvalid workers:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(workers.loc[workers.cnt_invalid_survey > 2,["hit_count","cnt_invalid_survey","mean_time"]])
    workers = workers.loc[(workers.cnt_invalid_survey <= 2)]
    
    # leave out those who didn't answer the survey
    print("\n\nWorkers who didn't answer the survey:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(workers.loc[(workers.hit_count > 0) & ((workers.w_own < 35) | (workers.w_own > 250) | (workers.h_own < 130) | (workers.h_own > 250) | (workers.gender == "none")),\
                        ["w_own", "h_own", "gender", "hit_count","cnt_invalid_survey","mean_time"]])
    workers = workers.loc[(workers.w_own >= 35) & (workers.w_own <= 250) & (workers.h_own >= 130) & (workers.h_own <= 250) & (workers.gender != "none")]
    
    print("\n\nValid workers who answered the survey:", workers.shape[0])
    
    print("Estimates from valid workers:")
    print(est.loc[est.workerId.isin(workers.index.values)].shape[0], "from", est.shape[0])
    est = est.loc[est.workerId.isin(workers.index.values)]
    
    return est, workers
    
def read_csv(files):
    
    dd = []
    for file in files:
        dd.append(pd.read_csv(file))
    data = pd.concat(dd)    
    
    hitsize = get_hitsize(data.columns)
    print("HITs of size:", hitsize)
    
    wmeas =["lbs","feet","inch"]
    wsurvey = ["age", "gender", "education"]
    est, images, workers, hits = init_datasets(data, wsurvey)
    
    iindex = 0
    # main loop of data processing -> iterate over each submission
    for index, row in data.iterrows():
        # read the survey answers (if any) and update the worker's information
        wid = row.WorkerId
        for s in wsurvey:
            if (row["Answer." + s] != "none"):
                workers.loc[wid,"cnt_invalid_survey"] += (workers.loc[wid,s] != "none" and workers.loc[wid,s] != row["Answer."+s])
                workers.loc[wid,s] = row["Answer."+s]
                
        for m in wmeas:
            if (row["Answer." + m] != "{}"): 
                if ("'" in row["Answer." + m]):
                    row["Answer." + m] = 5
                workers.loc[wid,"cnt_invalid_survey"] += (workers.loc[wid,m] != 0 and workers.loc[wid,m] != float(row["Answer."+m]))
                workers.loc[wid,m] = float(row["Answer."+m])
                
        workers.loc[wid,'hit_count'] += 1
        
        # parse time and answer sequences
        ts = row["Answer.timeseq"].split(',')[:-1]
        ts.append("0")
        ans = row["Answer.f0"].split(',')[:-1]
        
        if (len(ans) < hitsize):
            print(wid, "has not finished the task", row.HITId)
            continue
            
        # iterate through all guesses and add information to estimates
        for j in range(hitsize):
            im1, im2 = row["Input.im_left"+str(j)], row["Input.im_right"+str(j)]
            im1, im2 = im1.split("/")[-1], im2.split("/")[-1]
            im1, im2 = images.loc[im1], images.loc[im2]
            
            est.loc[iindex,["file1","file2"]] = [im1.name, im2.name]
            
            hnum = im1.num if im1.w_true < im2.w_true else im2.num 
            #hnum = im1.num if im1.h_true > im2.h_true else im2.num 
            
            est.loc[iindex,"num"] = ans[j]
            est.loc[iindex,"correct"] = 1 if ans[j] == hnum else 0
            
            est.loc[iindex,["workerId","hitId","assignmentId"]] = row.WorkerId, row.HITId, row.AssignmentId
            hits.loc[row.HITId,'filenames'].add((im1.name,im2.name))
            est.loc[iindex,"t"] = 1e-3*(float(ts[j+1]) - float(ts[j]))
            est.loc[iindex,"submitTime"] = float(ts[j+1])
            est.loc[iindex,"valid"] = 1
            iindex += 1
    
        print(iindex, "out of ", data.shape[0]*hitsize)
                
    workers["w_own"], workers["h_own"] = workers.lbs*0.453592, workers.feet*30.48 + workers.inch*2.54
    
    workers.to_csv(folderout + "/workers_all.csv", float_format='%.2f')
    est.to_csv(folderout + "/est_all.csv", float_format='%.2f')
    
    workers = pd.read_csv(folderout + "/workers_all.csv", index_col=0)
    est = pd.read_csv(folderout + "/est_all.csv", index_col=0)
    
    #est, workers = check_workers_validity(est, workers)    
    hits, workers = compute_HIT_bonus(est, images, workers, hits)
  
    # clean up data before saving
    est = est.loc[(est.valid > 0)]
    
    est.to_csv(folderout + "/est.csv", float_format='%.2f')
    workers.to_csv(folderout + "/workers.csv", float_format='%.2f')
    hits.to_csv(folderout + "/hits.csv", float_format='%.2f')
        
if __name__ == "__main__":
    
    if (len(sys.argv) < 3):
        print("Provide arguments: output_folder and a list with inputfiles in csv format")
        exit()
        
    folderout = sys.argv[1]
    files = sys.argv[2:]
    
    read_csv(files)

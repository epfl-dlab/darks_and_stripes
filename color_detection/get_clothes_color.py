import os
import sys
import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from matplotlib import rcParams

from model.cmu_model import get_testing_model

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

import scipy.signal as spsg

# fontsizes
fsticks = 16
fslegend = 16
fslabels = 20  

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
color = [100, 200, 0]

# definition of body line as (centered) 80% of the line between head and legs 
sf = 0.8

def cnn_forward_pass(oriImg, params, model_params):
    # on which scale to search, here single scale (=1) performs well enough (and is always faster)
    params['scale_search'] = [1]

    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    # calls to the CNN to create a Part Confidence Map (heatmap_avg) and Part Affinity Fields (paf_avg)
    # by averaging the outputs on several scales
    
    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])
                                                        
        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
        
        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)
    
    return heatmap_avg, paf_avg

def get_parts(heatmap_avg):
    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)
        
        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]
        
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    
    return all_peaks


def get_connections(oriImg, paf_avg, all_peaks):
    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    return connection_all, special_k

def get_poses(connection_all, special_k, all_peaks):
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    return subset, candidate

def get_body_line(index1, index2, candidate, canvas):
    
    if -1 in index1 and -1 in index2:
        return -np.ones(2), -np.ones(2)
    
    elif -1 in index2:
        X = candidate[index1.astype(int), 1]
        Y = candidate[index1.astype(int), 0]
    
    elif -1 in index1:
        X = candidate[index2.astype(int), 1]
        Y = candidate[index2.astype(int), 0]
        
    else:
        X = 0.5*(candidate[index1.astype(int), 1] + candidate[index2.astype(int), 1])
        Y = 0.5*(candidate[index1.astype(int), 0] + candidate[index2.astype(int), 0])    
    
    cur_canvas = canvas.copy()
    
    sX, fX = X[0]+.5*(1-sf)*(X[1]-X[0]), X[1]+.5*(1-sf)*(X[0]-X[1])
    sY, fY = Y[0]+.5*(1-sf)*(Y[1]-Y[0]), Y[1]+.5*(1-sf)*(Y[0]-Y[1])
     
    cv2.line(cur_canvas,(int(sY),int(sX)),(int(fY),int(fX)),color,5)
    plt.imshow(cur_canvas[:,:,[2,1,0]])
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return X, Y
   
def check_bounds(x, y, mx, my):
    return max(0, min(x, mx-1)), max(0, min(y, my-1)) 
   
def get_crossbody_line(X, Y, canvas):
    
    mX, mY = np.mean(X), np.mean(Y) 
    
    dX, dY = X[0] - mX, Y[0] - mY
    cX, cY = np.zeros(2), np.zeros(2)
    
    cX[0], cY[0] = round(mX+0.75*dY), round(mY-0.75*dX)
    cX[1], cY[1] = round(mX-0.75*dY), round(mY+0.75*dX)
    
    cX[0], cY[0] = check_bounds(cX[0], cY[0], canvas.shape[0], canvas.shape[1])
    cX[1], cY[1] = check_bounds(cX[1], cY[1], canvas.shape[0], canvas.shape[1])

    return cX, cY     
    
def sample_color_along_line(canvas, X, Y):    
    intcoef = np.linspace(0.5*(1.-sf), 1.-0.5*(1.-sf), int(max(np.abs(X[0]-X[1]), np.abs(Y[0]-Y[1]))))
        
    rcol, diff = np.zeros(3), []
    # sample pixels along the line and get their colors
    for alpha in intcoef:
        x, y = int(alpha*X[0] + (1.-alpha)*X[1]), int(alpha*Y[0] + (1.-alpha)*Y[1])
        rcol += canvas[x,y].astype(float)
        diff.append(canvas[x,y].astype(float))
    rcol /= len(intcoef)
    
    return rcol, diff
    
def plot_color_along_line(cur_canvas, diff, sdiff, fdiff):
    # Rescale the color from [0...255] to [0..1]
    cc = (color[0]/255, color[1]/255, color[2]/255)

    plt.plot(diff, color=cc)
    plt.ylabel('Brightness', fontsize=fslabels)
    plt.xlabel('Pixel index along main upper-body line', fontsize=fslabels)
    plt.xticks(fontsize=fsticks)
    plt.yticks(fontsize=fsticks)
    plt.ylim(-70, 120)
    plt.tight_layout()
    plt.savefig('../Plots/plot_stripes_colorline.pdf')
    plt.show()
    #plt.title("Pixel color (L2-norm) sampled along the line", fontsize=9)
    #plt.subplot(2,2,3)
    plt.plot(sdiff, color=cc)    
    plt.ylabel('Smoothed brightness', fontsize=fslabels)
    plt.xlabel('Pixel index along main upper-body line', fontsize=fslabels)
    plt.xticks(fontsize=fsticks)
    plt.yticks(fontsize=fsticks)
    plt.ylim(-70, 120)
    plt.tight_layout()
    plt.savefig('../Plots/plot_stripes_colorline_smooth.pdf')
    plt.show()
    
    #plt.title("Filtered (smoothed) pixel color", fontsize=9)
    
    #plt.subplot(2,2,4)
    plt.plot(fdiff[:40], color=cc)       
    plt.ylabel('Coefficient', fontsize=fslabels)
    plt.xlabel('Frequency', fontsize=fslabels)
    plt.xticks(fontsize=fsticks)
    plt.yticks(fontsize=fsticks)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig('../Plots/plot_stripes_colorline_fourier.pdf')
    plt.show()
    
    #plt.title("Fourier transform of smoothed color", fontsize=9)     
    #plt.subplot(2,2,1)
    plt.imshow(cur_canvas[:,:,[2,1,0]])
    plt.axis('off')
    #plt.title("Original image\nwith detected upper-body line", fontsize=9)
    #plt.tight_layout()
    plt.show()
    

def compute_body_color(n, subset, candidate, canvas):
    
    X, Y = get_body_line(subset[n][np.array(limbSeq[6]) - 1], subset[n][np.array(limbSeq[9]) - 1], candidate, canvas)
    if (X[0] < 0):
        return -1, -1
    #X, Y = get_crossbody_line(X, Y, canvas)
    
    rcol, diff = sample_color_along_line(canvas, X, Y)
    
    diff = np.linalg.norm(np.asarray(diff - rcol), axis=1)
    diff -= np.mean(diff)
    
    print(diff.shape)
    if (diff.shape[0] < 13):
        return -1, -1
    
    # compute filter size for smoothing (shouldn't be larger than a single stripes size!)            
    wl = round(np.abs(X[0]-X[1])/25)
    if (wl % 2 == 0):
        wl += 1
    wl = max(wl, 5)
    
    # smooth the signal    
    sdiff = spsg.medfilt(diff, 5)       
    sdiff = spsg.savgol_filter(sdiff, window_length=int(wl), polyorder=3)
    
    # real-valued signal -> rfft
    fdiff = np.fft.rfft(sdiff)
    fdiff = np.absolute(fdiff)
     
    plot_color_along_line(canvas, diff, sdiff, fdiff) 
     
    is_striped = 0
    
    s_max, s_min = np.amax(sdiff), np.amin(sdiff)
    f_max = np.amax(fdiff)
    f_max2_idx = 5+np.argmax(fdiff[5:max(26,len(fdiff))])
    f_max2 = fdiff[f_max2_idx]
    
    f_min_l, f_min_r = np.amin(fdiff[f_max2_idx-3:f_max2_idx+1]), np.amin(fdiff[f_max2_idx:f_max2_idx+4]) 
    
    # large change of color and a high pronounced peak in the range [5,25]
    if (s_max - s_min > 60 and f_max <= f_max2 and\
       1.5*max(f_min_l, f_min_r) <= f_max2 and max(fdiff[f_max2_idx-1], fdiff[f_max2_idx+1]) < f_max2):
        is_striped = 1

    print(rcol, is_striped)
    return rcol, is_striped


def process (input_image, params, model_params):
    
    oriImg = cv2.imread(input_image)  # B,G,R order
    
    while (oriImg.shape[0] > 1000 and oriImg.shape[1] > 1000):
        oriImg = cv2.resize(oriImg, (0,0), fx=0.5, fy=0.5)
    
    print(oriImg.shape)
    
    heatmap_avg, paf_avg = cnn_forward_pass(oriImg, params, model_params)
    
    all_peaks = get_parts(heatmap_avg)
    connection_all, special_k = get_connections(oriImg, paf_avg, all_peaks)
    
    subset, candidate = get_poses(connection_all, special_k, all_peaks)
    
    canvas = cv2.imread(input_image)  # B,G,R order
                            
    scores = subset[:,-2] / subset[:,-1] 
    print("Scores of detected people:", scores)
    
    idx = np.argsort(scores)
    
    if (len(scores) == 0):
        return []
        
    start_idx = np.sum(scores > scores[idx[-1]] - 0.15)
    print("Choose", start_idx, "most visible.")
    
    rv = []
    for i in range(len(idx)-start_idx, len(idx)):
        rv.append(compute_body_color(idx[i], subset, candidate, canvas))
    return rv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--processed_data', type=str, default='')

    # settings for plotting
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']
    rcParams['figure.autolayout'] = True

    args = parser.parse_args()
    keras_weights_file = args.model

    # load model
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()

    img_list = Path(args.folder).glob('*.jpg')
    if (args.processed_data != ""):
        img = pd.read_csv(args.processed_data, index_col=0)
    else:
        img = pd.DataFrame(columns=["r","g","b","stripes_auto"])
    
    img_list = ['../_all_/70G1dRn.jpg']#,'../_all_/3ikIhaL_2.jpg','../_all_/B42VUS6_2.jpg','../_all_/0dMOq2t_2.jpg']
    
    for img_file in img_list:
        # compute the color of the body
        print(str(img_file))
        img_name = str(img_file).split('/')[-1]
        #if (img_name in img.index.values):
        #    continue
        
        patterns = process(str(img_file), params, model_params)
        # ignore colors, focus on stripes
        img.loc[img_name,["r","g","b","stripes_auto"]] = [-1, -1, -1, -1]
        for p in patterns:
            img.loc[img_name,"stripes_auto"] = max(img.loc[img_name,"stripes_auto"], p[1])
        #cProfile.run('process(str(img_file), params, model_params)', 'stats')
        #img.to_csv(args.folder+"data_cd.csv", float_format="%.2f")

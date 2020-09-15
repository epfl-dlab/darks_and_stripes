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
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

from routines import *
from matching import *

# Option1: analysis of the big "dirty" data set for color   
def read_dirty_data(folder, stripes = [0,1], colors=["dark","light","none"]):    
    est, images, workers = pd.read_csv(folder + "/est.csv", index_col = 0),\
                           pd.read_csv(folder + "/images_final.csv", index_col = "filename"),\
                           pd.read_csv(folder + "/workers.csv", index_col = "workerId")
                         
    images = images.loc[images.naked == 0]
    images = images.loc[images.gender.isin(["female","male"])]
    images = images.loc[images.w_true > 30]
    images = images.loc[images.h_true > 100]  
        
    print(images.loc[images.gender == "female"].shape[0])
    
    print(images.shape[0])
    assign_BMI_categories(images)
    
    print("Light, dark, neutral:")
    print(images.loc[images.color_type == "light"].shape[0], images.loc[images.color_type == "dark"].shape[0], images.shape[0])
    
    print("Striped:")
    print(images.loc[images.stripes > 0].shape[0])
    
    images.color_type = [str(x) for x in images.color_type.values]
    images.loc[images.color_type=="nan","color_type"] = "none" 
    
    images = images.loc[(images.stripes.isin(stripes)) | images.color_type.isin(colors)]
    
    print(est.shape, images.shape, workers.shape)
    est = est.loc[est.filename.isin(images.index.values)]
    workers = workers.loc[np.unique(est.workerId)]
    print(est.shape, images.shape, workers.shape)

    return est, images, workers
      
    
def read_clean_data(folder1, folder2):
    est1, images1, workers1 = pd.read_csv(folder1 + "/est.csv", index_col = 0),\
                              pd.read_csv(folder1 + "/images.csv", index_col = "filename"),\
                              pd.read_csv(folder1 + "/workers.csv", index_col = "workerId")
                           
    images1["color_type"] = folder1.split("/")[1].split("_")[-1]
    
    est1 = est1.loc[est1.filename.isin(images1.index.values)]
    workers1 = workers1.loc[np.unique(est1.workerId)]
    
    print("Total number of images:", images1.shape[0])
    print("Total number of estimates:", est1.shape[0])
    print("Total number of workers:", workers1.shape[0])
       
    est2, images2, workers2 = pd.read_csv(folder2 + "/est.csv", index_col = 0),\
                              pd.read_csv(folder2 + "/images.csv", index_col = "filename"),\
                              pd.read_csv(folder2 + "/workers.csv", index_col = "workerId")


    images2["color_type"] = folder2.split("/")[1].split("_")[-1]
    #print(images2["color_type"])
    
    workers1 = workers1.loc[workers1.country == "United States"]
    est1 = est1.loc[est1.workerId.isin(workers1.index.values)]
    
    workers2 = workers2.loc[workers2.country == "United States"]
    est2 = est2.loc[est2.workerId.isin(workers2.index.values)]
    
    print(est1.shape, est2.shape)
    
    est, images, workers = pd.concat([est1, est2], join='outer'),\
                           pd.concat([images1, images2], join='outer'), pd.concat([workers1, workers2], join='outer')
                                 
    est = est.loc[est.filename.isin(images.index.values)]
    est_cnt = est.groupby('filename').count()
    print(min(est_cnt['workerId']), max(est_cnt['workerId']))
    return est, images, workers
    
def dist_plot(imgt, imgc, trt_name, ctr_name, meas, qname, unit, xdelta, binstep):
    if (unit != ""):
        unit = " [" + unit + "]"
    
    h = np.linspace(min(np.amin(imgt[meas+"_true"]), np.amin(imgc[meas+"_true"])), max(np.amax(imgt[meas+"_true"]), np.amax(imgc[meas+"_true"])), 11)
    
    print("Treated vs control for ", meas, ":")
    print(">:", np.sum(imgt[meas+"_est"].values > imgc[meas+"_est"].values))
    print("<:", np.sum(imgt[meas+"_est"].values < imgc[meas+"_est"].values))
    
    sw, pval = spst.wilcoxon(imgt[meas+"_err"], imgc[meas+"_err"])
    print("Pairwise difference:", trt_name, "-", ctr_name, "for", meas, unit)
    print('Wilcoxon (t, pval): %.3lf, %.5lf' % (sw, pval))
    print("Treated:", np.mean(imgt[meas+"_est"]), unit, "vs untreated:", np.mean(imgc[meas+"_est"]), unit)
    
    print("Effect strength:", bs.bootstrap(imgt[meas+"_est"].values - imgc[meas+"_est"].values,\
                              stat_func=bs_stats.mean, alpha=0.05, num_iterations=10000))
    
    imgt[meas+"_rel_diff"] = 2*(imgt[meas+"_est"].values - imgc[meas+"_est"].values) / (imgt[meas+"_true"].values + imgc[meas+"_true"].values)
    sw, pval = spst.wilcoxon(imgt[meas+"_rel_diff"])
    print("\nRelative pairwise difference:", trt_name, "-", ctr_name, "for", meas, '%')
    print('Wilcoxon (t, pval): %.3lf, %.5lf' % (sw, pval))
    print("Effect strength:", bs.bootstrap(imgt[meas+"_rel_diff"].values, stat_func=bs_stats.mean, alpha=0.05, num_iterations=10000))
    
    constant_bins = range(-xdelta, xdelta, binstep)
    sns.distplot(imgt[meas+"_est"].values-imgc[meas+"_est"].values, bins=constant_bins, color=get_meas_color(meas)[0])
    plt.axvline(x=0, color="black", linewidth='1.0', linestyle="dashed")
    plt.xlim(-xdelta, xdelta)
    plt.xlabel("Estimated within-pair %s diff.%s: %s minus %s" % (qname, unit, trt_name, ctr_name), fontsize = fslegend)
    plt.ylabel('Relative frequency', fontsize = fslegend)
    plt.xticks(fontsize = fsticks)
    plt.yticks(fontsize = fsticks)
    plt.legend(fontsize = fslegend)
    plt.tight_layout()
    plt.savefig('Plots/' + 'plot_dist_' + meas + '_pairwise_est_clean_' + trt_name + '_' + ctr_name + '.pdf')
    plt.show()   
       
def scatter_plot(imgt, imgc, trt_name, ctr_name, meas, qname, unit, mmin, mmax):       
    #plot sample vs sample with confidence intervals
    pltsym = ['p','.','*']
    for bt in np.unique(imgt.bmitype.values):
        #bc = get_BMI_color(np.unique(imgt.loc[imgt.bmitype==bt,"bmiclass"])[0])
        bc = np.unique(imgt.loc[imgt.bmitype==bt,"bmiclass"])[0]
        select_idx = (imgt.bmitype.values == bt)
        imgtc, imgcc = imgt.loc[select_idx], imgc.loc[select_idx]
        plt.scatter(imgtc[meas+"_est"], imgcc[meas+"_est"], color=get_meas_color(meas)[bc], label=bt, marker = pltsym[bc])
    
    handles,labels = plt.gca().get_legend_handles_labels()

    handles = [handles[0], handles[2], handles[1]]
    labels = [labels[0], labels[2], labels[1]]

    plt.gca().legend(handles, labels, fontsize=fslegend)
    
    #for i in range(imgt.shape[0]):
    #    plt.text(imgt[meas+"_err"].values[i], imgc[meas+"_err"].values[i],\
    #     imgt.index.values[i], fontsize=8)
    #    imgt.index.values[i] + "\n" + imgc.index.values[i], fontsize=8)   
    
    plt.plot([mmin, mmax], [mmin, mmax], color="black", linewidth='1.0', linestyle="dashed")
             #[np.amin(imgt[meas+"_est"].values-.5),np.amax(imgt[meas+"_est"].values+.5)],\          
    plt.xlim(mmin, mmax)
    plt.ylim(mmin, mmax)

    if (unit != ""):
        unit = " [" + unit + "]"
    
    plt.xlabel("Mean %s estimate for %s%s" % (qname, trt_name, unit), fontsize=fslegend)
    plt.ylabel("Mean %s estimate for %s%s" % (qname, ctr_name, unit), fontsize=fslegend)
    plt.xticks(fontsize=fsticks)
    plt.yticks(fontsize=fsticks)
    
    plt.gca().set_aspect(aspect=1)
    plt.tight_layout()
    plt.savefig('Plots/plot_scatter_' + meas + '_pairwise_comp_clean_' + trt_name + '_' + ctr_name + '.pdf')
    plt.show()
        
# plot treated vs control 
def plot_matched_results(imgt, imgc, trt_name, ctr_name, est, workers):    

    print("Matching:", imgt.shape[0], trt_name, "vs", imgc.shape[0], ctr_name, "samples.")
    
    print("Wilcoxon's test, mean diff, mean std. diff.:")
    
    # Wilcoxon's rank test
    vals = imgt["w_true"].values - imgc["w_true"].values
    if sum(vals > 0.01):
        sw, pval = spst.wilcoxon(vals)
        print("Weight: %.4lf, %.4lf (), %.4lf" % (pval, np.mean(vals), np.std(vals)))
        print("95% CI for the mean difference: ", bs.bootstrap(vals, stat_func=bs_stats.mean, alpha=0.05, num_iterations=10000))
    
    vals = imgt["h_true"].values - imgc["h_true"].values
    if sum(vals > 0.01):
        sw, pval = spst.wilcoxon(vals)
        print("Height: %.4lf, %.4lf, %.4lf" % (pval, np.mean(vals), np.std(vals)))
        print("95% CI for the mean difference: ", bs.bootstrap(vals, stat_func=bs_stats.mean, alpha=0.05, num_iterations=10000))
    
    # for the chosen border (2.5cm) all height estimates match exactly -> p-val = 1

    vals = imgt["bmi_true"].values - imgc["bmi_true"].values
    if sum(vals > 0.01):
        sw, pval = spst.wilcoxon(vals)
        print("BMI: %.4lf, %.4lf, %.4lf" % (pval, np.mean(vals), np.std(vals)))
        print("95% CI for the mean difference: ", bs.bootstrap(vals, stat_func=bs_stats.mean, alpha=0.05, num_iterations=10000))
    
    
    print("Gender in", trt_name+":", imgt.loc[imgt.gender=="female"].shape[0], " female,", imgt.loc[imgt.gender=="male"].shape[0], "male")
    print("Gender in", ctr_name+":", imgc.loc[imgc.gender=="female"].shape[0], " female,", imgc.loc[imgc.gender=="male"].shape[0], "male")
    
    # simple plot with error distributions
    dist_plot(imgt, imgc, trt_name, ctr_name, 'w', 'weight', 'kg', 30, 3)
    dist_plot(imgt, imgc, trt_name, ctr_name, 'h', 'height', 'cm', 12, 1)
    dist_plot(imgt, imgc, trt_name, ctr_name, 'bmi', 'BMI', '', 11, 1)
    
    # plot sample vs sample (use matching)
    scatter_plot(imgt, imgc, trt_name, ctr_name, 'w', 'weight', 'kg', 49, 140)
    scatter_plot(imgt, imgc, trt_name, ctr_name, 'h', 'height', 'cm', 157, 185)    
    scatter_plot(imgt, imgc, trt_name, ctr_name, 'bmi', 'BMI', '', 17, 40)
    
    # print 5 couples with the largest difference between the groups    
    imgt = imgt.sort_values(by = 'w_est_diff')
    print(imgt[["w_est","w_est2","paired_filename"]])
    idmax = imgt.index.values[:10]
    idmin = imgt.index.values[-10:]
    
    print([*idmax, *idmin])
    
    print(trt_name, "most underestimated compared to", ctr_name)
    print(imgt.loc[idmin,["paired_filename","w_err_diff","w_true","w_true2","reddit_id"]])
    print(trt_name, "most overestimated compared to", ctr_name)
    print(imgt.loc[idmax,["paired_filename","w_err_diff","w_true","w_true2"]])
    
    for (num, i) in enumerate([*idmin, *idmax]):        
        plt.subplot(1,2,1)
        plt.imshow(mpimg.imread('_all_/'+i))
        plt.title("%s %s\nTrue: %.2lf kg, %.2lf cm\nEst: %.2lf kh, %2.lf cm" %\
            (i, imgt.loc[i,"reddit_id"], imgt.loc[i,"w_true"], imgt.loc[i,"h_true"], imgt.loc[i,"w_est"], imgt.loc[i,"h_est"]))
        plt.axis('off')
        plt.subplot(1,2,2)
        j = imgt.loc[i,"paired_filename"]
        plt.imshow(mpimg.imread('_all_/'+j))
        plt.title("%s %s\nTrue: %.2lf kg, %.2lf cm\nEst: %.2lf kh, %2.lf cm" %\
            (j, imgc.loc[j,"reddit_id"], imgc.loc[j,"w_true"], imgc.loc[j,"h_true"], imgc.loc[j,"w_est"], imgc.loc[j,"h_est"]))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('Plots/samplewise' + str(num) + '.pdf')
        #plt.show()
    
    
        cesti, cestj = est.loc[est.filename == i], est.loc[est.filename == j]
        plt.subplot(2,1,1)
        plt.title("Weight estimates")
        sns.distplot(cesti.w, label=(trt_name + " " + str(np.mean(cesti.w))) )
        sns.distplot(cestj.w, label=(ctr_name + " " + str(np.mean(cestj.w))) )
        plt.legend()
        
        plt.subplot(2,1,2)
        plt.title("Workers weight")
        sns.distplot(workers.loc[cesti.workerId.values,"w_own"], label=trt_name)
        sns.distplot(workers.loc[cestj.workerId.values,"w_own"], label=ctr_name)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('Plots/samplewise' + str(num) + "_dist.pdf")
        #plt.show()
    
def plot_image(im):    
    print(im.name)
    plt.imshow(mpimg.imread('_all_/'+im.name))
    plt.title("True: %.2lf cm, %.2lf kg\nEst.: %.2lf cm %.2lf kg" % (im.h_true, im.w_true, im.h_est, im.w_est), fontsize=5)
    plt.axis('off')
    
def show_matching_pairs(imgt, imgc):
    
    mpt = set()#build_est_matching_pairs(imgt)
    mpc = build_est_matching_pairs(imgc)
    pairs = mpt.union(mpc)
    
    img = pd.concat([imgt, imgc])
    
    print("Number of matching pairs:", len(pairs))

    pdd = {"img1":[], "img2":[]}
    for p in pairs:
        # build pairs of opposite color
        '''
        if ("_light" in p[0]):
            op = [x.replace("_light","_dark") for x in p]
        else:
            op = [x.replace("_dark","_light") for x in p]
        '''
        # add striped pair 
        op = [x.replace("_light", "") for x in p]
        # replace _light with _dark at random
        if (np.random.rand() > 0.5):
            p = (p[0].replace("_light", "_dark"), p[1].replace("_light","_dark"))
                
        # add 4 possible combinations of pairs    
        pdd["img1"].append(p[0].replace(".jpg", "_numbered.jpg"))
        pdd["img2"].append(p[1].replace(".jpg", "_numbered.jpg"))
        
        pdd["img1"].append(p[0].replace(".jpg", "_numbered.jpg"))
        pdd["img2"].append(op[1].replace(".jpg", "_numbered.jpg"))
        
        pdd["img1"].append(op[0].replace(".jpg", "_numbered.jpg"))
        pdd["img2"].append(p[1].replace(".jpg", "_numbered.jpg"))
        
        pdd["img1"].append(op[0].replace(".jpg", "_numbered.jpg"))
        pdd["img2"].append(op[1].replace(".jpg", "_numbered.jpg"))
        
        
    pid = pd.DataFrame(data=pdd)
    #pid.to_csv("matched_pairs.csv")
            
if __name__ == "__main__":

    _, im, w = read_dirty_data('large_set_results', stripes=[], colors=["light", "dark"])
    
    if (len(sys.argv) == 2):
        est, images, workers = read_dirty_data(sys.argv[1], stripes=[1], colors=["light"])
        images = images.loc[images.color_type == "light"]
        #simages = images.loc[(images.stripes) > 0]
    else:
        est, images, workers = read_clean_data(sys.argv[1], sys.argv[2])
        images = compute_image_error(est, images)
        simages = images.loc[(images.color_type) == sys.argv[1].split("/")[1].split("_")[-1]]
        #print(np.mean(images.loc[images.type==0,"h_err_u"]), np.mean(images.loc[images.type == 1,"h_err_u"]))

    # plot settings
    sns.set(style="whitegrid", palette="bright", color_codes=True)
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']
    rcParams['figure.autolayout'] = True
    
    # set the random seed to make results reproducable
    np.random.seed(239)
        
    # Return format: subset_of_simages, matched_images
    #imgt, imgc = match_on_covariates(images, simages, bound=2.5)
    imgt, imgc = max_match_on_covariates(images, simages, bound=2.5)#0.01)
    
    imgt, imgc = images.loc[imgt], images.loc[imgc] 

    print(imgt.shape[0], imgt.loc[imgt.gender == "female"].shape[0], imgt.loc[imgt.gender == "male"].shape[0])
     
    print("Number of estimates >=", np.amin(est.loc[est.filename.isin(imgc.index.values)].groupby('filename').apply(lambda x: x.shape[0]))) 
     
    imgt, imgc = assign_BMI_categories(imgt), assign_BMI_categories(imgc)
    
    imgt["w_err_diff"] = imgt.w_err.values - imgc.w_err.values
    imgt["w_true2"] = imgc.w_true.values
    imgt["paired_filename"] = imgc.index.values
    imgt["w_est2"] = imgc.w_est.values
    imgt["w_est_diff"] = imgt.w_est - imgt.w_est2
         
    valid = [idx.replace("_light","").replace("_dark","") != x.paired_filename.replace("_dark","").replace("_light","")\
             for idx, x in imgt.iterrows()]
    print("Wrongly assigned:", imgt.loc[np.asarray(valid),"paired_filename"])
     
    # visualize the results
    plot_matched_results(imgt, imgc, "light", "striped", est, workers)
    
    # m-groups comparison
    #for m in [1,2,3,5,10,15,20]:
    #    print(str(m)+":", "%.5lf, %.3lf, %.3lf " % (m_group_effects(m, imgt.bmi_err.values, imgc.bmi_err.values)))
    
    #print(imgc.shape)
    #show_matching_pairs(imgt, imgc)
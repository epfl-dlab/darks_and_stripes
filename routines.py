import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# fontsizes
fsticks = 16
fslegend = 16
fslabels = 20      

def get_BMI_color(bmiclass):
    colors = ['r','g','b']
    return colors[bmiclass]

def get_meas_color(meas):
    if meas == "w" or meas == "weight":
        return ["royalblue", "lightskyblue", "navy"]
    elif meas == "h" or meas == "height":
        return ["sandybrown", "navajowhite", "sienna"]   
    else:
        return ["grey", "lightgray", "k"]
        
        
def bootstrap_ci_estimate(x, conf, p=1000):
    sx = np.zeros(p)
    for i in range(p):
        sx[i] = np.mean(np.random.choice(x, x.shape[0], replace=True))
    sx = np.sort(sx)
    print(len(x))
    return sx[round(0.5*(1-conf)*p)], sx[round((1-0.5*(1-conf))*p)]
        
def compute_image_error(est, images, conf=0.95):
    # weight/height estimates and errors
    images["w_est"], images["h_est"] = est.groupby("filename").apply(lambda x: np.mean(x.w)),\
                                       est.groupby("filename").apply(lambda x: np.mean(x.h))
    images["w_dev"], images["h_dev"] = est.groupby("filename").apply(lambda x: np.std(x.w)),\
                                       est.groupby("filename").apply(lambda x: np.std(x.h))
    
    images["w_err"] = images.w_true - images.w_est
    images["h_err"] = images.h_true - images.h_est
    
    # BMI + BMI-errors
    est["bmi"] = 10000.*est["w"] / ((est["h"])**2)
    images["bmi_true"] = 10000.*images["w_true"] / ((images["h_true"])**2)
    images["bmi_est"] = est.groupby("filename").apply(lambda x: np.mean(x.bmi))
    images["bmi_err"] = images.bmi_true - images.bmi_est
    '''
    # bootstrap for CI of weight, height and BMI estimates 
    images[["w_est_l","w_est_u"]] = est.groupby("filename").apply(lambda x: bootstrap_ci_estimate(x.w, conf)).apply(pd.Series)    
    images["w_err_l"], images["w_err_u"] = images.w_true - images.w_est_u - images.w_err, images.w_true - images.w_est_l - images.w_err
    
    images[["h_est_l","h_est_u"]] = est.groupby("filename").apply(lambda x: bootstrap_ci_estimate(x.h, conf)).apply(pd.Series)
    images["h_err_l"], images["h_err_u"] = images.h_true - images.h_est_u - images.h_err, images.h_true - images.h_est_l - images.h_err
    
    images[["bmi_est_l","bmi_est_u"]] = est.groupby("filename").apply(lambda x: bootstrap_ci_estimate(x.bmi, conf)).apply(pd.Series)    
    images["bmi_err_l"], images["bmi_err_u"] = images.bmi_true - images.bmi_est_u - images.bmi_err, images.bmi_true - images.bmi_est_l - images.bmi_err
    '''
    return images
    
def assign_BMI_categories(img):
    bmival = np.sort(np.asarray(img.bmi_true.values))
    n = len(bmival)
    
    img["bmitype"] = "obese"# $>30$"
    img["bmiclass"] = 2
    
    img.loc[img.bmi_true <= 30,"bmitype"] = "overweight"# $25-30$"
    img.loc[img.bmi_true <= 30,"bmiclass"] = 0    
    
    img.loc[img.bmi_true <= 25,"bmitype"] = "normal"# $18.5-25$"
    img.loc[img.bmi_true <= 25,"bmiclass"] = 1    
    
    img.loc[img.bmi_true <= 18.5,"bmitype"] = "underweight"# $\leq 18.5$"
    img.loc[img.bmi_true <= 18.5,"bmiclass"] = 1    
    
    return img
    
    
def bootstrap_ci_series(x, n, conf, p=1000):
    s = np.zeros((p,n))
    for i in range(p):
        s[i,:] = np.random.choice(x, n, replace=True)
        print(s[i,:])
        for j in range(n-1,0,-1):
            s[i,j] = np.mean(s[i,:(j+1)])
            
    for i in range(n):
        s[:,i] = np.sort(s[:,i])
        print(s[0,i], s[p-1,i])
    return s[round(0.5*(1-conf)*p),:], s[round((1-0.5*(1-conf))*p),:]
        
    
def plot_time_evolution(im0, im1, est):
    
    est0 = est.loc[est.filename == im0]
    est1 = est.loc[est.filename == im1]
    n = min(est0.shape[0], est1.shape[0])
    
    e0min, e0max = bootstrap_ci_series(est0.w.values, n, 0.95)
    e1min, e1max = bootstrap_ci_series(est1.w.values, n, 0.95)
    
    plt.plot(range(n), e0min, color='red', label=im0)
    plt.plot(range(n), e0max, color='red', label=im0)
    
    plt.plot(range(n), e1min, color='blue', label=im1)
    plt.plot(range(n), e1max, color='blue', label=im1)
    
    plt.legend()
    
    plt.show()
    plt.clf()
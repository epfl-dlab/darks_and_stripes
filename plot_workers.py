import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import process_tc as pt

def get_quantiles(values):
	stats = {}
	stats["mean"], stats["median"] = np.mean(values), np.median(values)
	stats["q25"], stats["q75"] = np.percentile(values, [25, 75])
	return stats

def get_counts(values, categories):
	counts = {}
	for c in categories:
		counts[c] = np.sum(values == c) / len(values)
	return counts

_, _, w_large = pt.read_dirty_data('large_set_results', stripes=[1], colors=["dark", "light"])
_, _, w1 = pt.read_clean_data('clean_set_results/res_clothes_dark', 'clean_set_results/res_clothes_light')
_, _, w2 = pt.read_clean_data('clean_set_results/res_clothes_stripes', 'clean_set_results/res_clothes_stripes')

w  = pd.concat([w_large, w1, w2], join='outer')
w = w.groupby(w.index).first()

w["bmi_own"] = 10000.*w["w_own"] / ((w["h_own"])**2)

print(w.shape)
print("Weight:", get_quantiles(w.w_own))
print("Height:", get_quantiles(w.h_own))
print("BMI:", get_quantiles(w.bmi_own))

print("Countries:", get_counts(w.country, ["United States", "India"]))
print("Age:", get_counts(w.age, np.unique(w.age)))
print("Education:", get_counts(w.education, np.unique(w.education)))
w = w.loc[w.gender.isin(["male", "female"])]
print("Gender:", get_counts(w.gender, np.unique(w.gender)))

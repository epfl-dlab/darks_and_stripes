import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

allcost = pd.read_csv('costs.csv')
print(allcost.columns.values)

ids = np.asarray([])

curdirectory = os.path.dirname(__file__)
directory = os.path.join(curdirectory, 'mturk_results')

for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith('.csv'):
           #  perform calculation
           costs = pd.read_csv(root + os.sep + file)
           if ('AssignmentId' in costs.columns.values):
               ids = np.append(ids, costs.AssignmentId)

allcost = allcost.loc[allcost.AssignmentID.isin(ids)]

allcost.DateInitiated = [(x.DateInitiated.split(' ')[0]).split('/')[0] for _, x in allcost.iterrows()]

print(np.sum(allcost.Amount))
print(0.25*np.sum(allcost.Amount))

print(np.sum(allcost.loc[allcost.DateInitiated == '1'].Amount)*1.25)
print(np.sum(allcost.loc[allcost.DateInitiated == '2'].Amount)*1.25)
print(np.sum(allcost.loc[allcost.DateInitiated == '3'].Amount)*1.25)




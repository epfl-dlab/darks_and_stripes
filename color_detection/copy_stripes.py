import pandas as pd
from shutil import copyfile

images = pd.read_csv('mturk_res/images_proc_cd.csv', index_col='filename')
images = images.loc[images.stripes_auto > 0]
print(images.shape)

for img in images.index.values:
    try: 
        copyfile('../fashion_detection/_all_/'+img, '../fashion_detection/_stripes_auto_/'+img)
    except FileNotFoundError:
        print("File not found: was considered naked by the new policy :)")
        
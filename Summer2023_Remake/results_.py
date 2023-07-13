import pickle as pk
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np

def results_now(weights, pd_frames):
    results = {x :[] for x in ['HRP', 'TBHRP', 'IV', 'EQ','MV']}
    n = 30
    for tpe in ['HRP', 'TBHRP', 'IV', 'EQ','MV']:
        for x in weights:
            x['EQ'] = pd.Series([1/n for x in range(n)], index = x['HRP'].index)
        pd_frames_ = pd_frames.copy()
        for loop in range(len(pd_frames_)-1):
            ### First set in right order
            pd_frames_[loop+1] = pd_frames_[loop+1][weights[loop][tpe].index]

            ### Add one
            pd_frames_[loop+1] += 1

            ### Mutiply first row
            pd_frames_[loop+1].iloc[0] *= weights[loop][tpe]

            ### Final Return
            results[tpe].append(pd_frames_[loop+1].cumprod(axis = 0).iloc[-1].sum()-1)
    return results

home_dir_frames = r"C:\Users\blake\Downloads\pd_frames\\"
home_dir_weights = r"C:\Users\blake\Downloads\weights\\"


onlyfiles_frame = [f for f in listdir(home_dir_frames) if isfile(join(home_dir_frames, f))]
onlyfiles_weights = [f for f in listdir(home_dir_weights) if isfile(join(home_dir_weights, f))]
### Load files here\
for f, w in zip(onlyfiles_frame,onlyfiles_weights):
    with open(home_dir_frames + f, 'rb') as _:
        frame = pk.load(_)
    with open(home_dir_weights + w, 'rb') as _:
        weights = pk.load(_)
    rs = results_now(weights, frame)
    if len(rs['TBHRP']) > 0:
        print(f)
        print(w)
        print(np.mean(np.array(rs['TBHRP'])))
        print(np.std(np.array(rs['TBHRP'])))
        print(np.mean(np.array(rs['TBHRP']))/np.std(np.array(rs['TBHRP'])))


'''
plt.legend()
plt.show()

print(type)
print(results)
np.cumprod(np.array(results) + 1)[-1]-1

    
###Potential Speed up
n = len(df)/30
np.array_split(df, 30)
'''
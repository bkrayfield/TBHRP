import os
import pickle
import numpy as np
import pandas as pd

def calculate_results(weights, pd_frames):
    result_categories = ['HRP', 'TBHRP', 'IV', 'EQ','MV']
    results = {category: [] for category in result_categories}
    num_results = 30

    for category in result_categories:
        for weight in weights:
            weight['EQ'] = pd.Series([1/num_results for _ in range(num_results)], index=weight['HRP'].index)

        pd_frames_copy = pd_frames.copy()

        for index in range(len(pd_frames_copy)-1):
            # Set in right order
            pd_frames_copy[index+1] = pd_frames_copy[index+1][weights[index][category].index]

            # Add one to DataFrame
            pd_frames_copy[index+1] += 1

            # Multiply first row
            pd_frames_copy[index+1].iloc[0] *= weights[index][category]

            # Final Return
            results[category].append(pd_frames_copy[index+1].cumprod(axis=0).iloc[-1].sum() - 1)
    
    return results

home_dir_frames = r"C:\Users\blake\Downloads\pd_frames\\"
home_dir_weights = r"C:\Users\blake\Downloads\weights\\"

frames_files = [file for file in os.listdir(home_dir_frames) if os.path.isfile(os.path.join(home_dir_frames, file))]
weights_files = [file for file in os.listdir(home_dir_weights) if os.path.isfile(os.path.join(home_dir_weights, file))]

for frame_file, weight_file in zip(frames_files, weights_files):
    with open(os.path.join(home_dir_frames, frame_file), 'rb') as file:
        frame = pickle.load(file)
    with open(os.path.join(home_dir_weights, weight_file), 'rb') as file:
        weights = pickle.load(file)

    results = calculate_results(weights, frame)

    if results['TBHRP']:
        print(frame_file)
        print(weight_file)
        tbhrp_results = np.array(results['TBHRP'])
        print(np.mean(tbhrp_results))
        print(np.std(tbhrp_results))
        print(np.mean(tbhrp_results) / np.std(tbhrp_results))

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
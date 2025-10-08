#### This is a test file for a revision

import pandas as pd

# Read the CSV file
df = pd.read_csv(r"C:\Users\n00812642\OneDrive - University of North Florida\Research\TBHRP\Revision 1\HRP\top500Total.csv")
itit__ = getFilingURLSin(df.columns, API_KEY)


# Extract 'idx' column as date vector
date_vector = df[['idx']]

# Compute log returns
return_ = np.log(df[list(df.columns)[1:]]) - np.log(df[list(df.columns)[1:]].shift(1))
keep_list = list(set([x for x in itit__.ticker.tolist() if x in df.columns]))
return_.index = date_vector
return_ = return_[keep_list]


# Update the dataframe with log returns and set index
df = return_.copy()

mw = pd.read_csv(r"C:\Users\n00812642\Downloads\Market_Weights.csv")

mw.head()

mw.dropna(inplace = True)

### 
T50 = mw.iloc[:50].Ticker



# Get the tickers from columns
TICKERS = T50.to_list()
TICKERS = [x for x in TICKERS if x != "ANTM"]
TICKERS = [x for x in TICKERS if x != "GOOGL"]
TICKERS = [x for x in TICKERS if x not in ['GPS', 'ADS', 'CTL', 'COG', 'RE', 'ABC', 'UAA', 'PKI','LB','PEAK']]
df = df[TICKERS]
df.dropna(inplace = True)



# Generate SIMMAT using API_KEY and YEARS
YEARS_TEXT = [YEARS[0]- 1, YEARS[1]]

# Generate SIMMAT using API_KEY and YEARS
keep = GenerateSIMMAT(API_KEY, TICKERS, YEARS_TEXT)
keep, unclean = keep.create_simmat()

df.index = pd.to_datetime([x[0] for x in df.index])

df = df[(df.index >= unclean.index.min()) & (df.index <= unclean.index.max())]
n = 50
list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]



total_save = []
for frame_num in list_df:

    ### Get max date and find the right matrix for the optimization
    max_date = frame_num.index.max()
    keep_unclean = unclean[(unclean.index <= max_date)]
    keep_unclean = keep_unclean[(keep_unclean.index == keep_unclean.index.max())]
    
    total_save.append(final_fun(frame_num, TICKERS, keep_unclean))

import pickle as pk

OUTPUT_DIR = "C:\\Users\\n00812642\\Desktop\\Results_TBHRP\\"

with open(OUTPUT_DIR + rnum + "T50_frameswreturns.pk",'wb') as file_:
    pk.dump(list_df, file_)
with open(OUTPUT_DIR + rnum + "T50weights.pk",'wb') as file_:
    pk.dump(total_save, file_)



######################################## Bottom ###############################

#### This is a test file for a revision

import pandas as pd

# Read the CSV file
df = pd.read_csv(r"C:\Users\n00812642\OneDrive - University of North Florida\Research\TBHRP\Revision 1\HRP\top500Total.csv")
itit__ = getFilingURLSin(df.columns, API_KEY)


# Extract 'idx' column as date vector
date_vector = df[['idx']]

# Compute log returns
return_ = np.log(df[list(df.columns)[1:]]) - np.log(df[list(df.columns)[1:]].shift(1))
keep_list = list(set([x for x in itit__.ticker.tolist() if x in df.columns]))
return_.index = date_vector
return_ = return_[keep_list]


# Update the dataframe with log returns and set index
df = return_.copy()

mw = pd.read_csv(r"C:\Users\n00812642\Downloads\Market_Weights.csv")

mw.head()

mw.dropna(inplace = True)

### 
B50 = mw.iloc[-50:].Ticker


# Get the tickers from columns
TICKERS = B50.to_list()

df = df[TICKERS]
df.dropna(inplace = True)



# Generate SIMMAT using API_KEY and YEARS
YEARS_TEXT = [YEARS[0]- 1, YEARS[1]]

# Generate SIMMAT using API_KEY and YEARS
keep = GenerateSIMMAT(API_KEY, TICKERS, YEARS_TEXT)
keep, unclean = keep.create_simmat()

df.index = pd.to_datetime([x[0] for x in df.index])

df = df[(df.index >= unclean.index.min()) & (df.index <= unclean.index.max())]
n = 50
list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]



total_save = []
for frame_num in list_df:

    ### Get max date and find the right matrix for the optimization
    max_date = frame_num.index.max()
    keep_unclean = unclean[(unclean.index <= max_date)]
    keep_unclean = keep_unclean[(keep_unclean.index == keep_unclean.index.max())]
    
    total_save.append(final_fun(frame_num, TICKERS, keep_unclean))

import pickle as pk

OUTPUT_DIR = "C:\\Users\\n00812642\\Desktop\\Results_TBHRP\\"

with open(OUTPUT_DIR + rnum + "B50_frameswreturns.pk",'wb') as file_:
    pk.dump(list_df, file_)
with open(OUTPUT_DIR + rnum + "B50weights.pk",'wb') as file_:
    pk.dump(total_save, file_)






####################################################################################################
#                                   Results                                                        #
####################################################################################################

import os
import pickle
import numpy as np
import pandas as pd

def calculate_results(weights, pd_frames):
    result_categories = ['HRP', 'TBHRP', 'IV', 'EQ','MV']
    results = {category: [] for category in result_categories}
    num_results = len(weights[0]['HRP'])

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

home_dir_frames = "C:\\Users\\n00812642\\Desktop\\Results_TBHRP\\"
home_dir_weights = "C:\\Users\\n00812642\\Desktop\\Results_TBHRP\\"

frames_files = [file for file in os.listdir(home_dir_frames) if os.path.isfile(os.path.join(home_dir_frames, file))]
weights_files = [file for file in os.listdir(home_dir_weights) if os.path.isfile(os.path.join(home_dir_weights, file))]

frames_files = [x for x in frames_files if x.find("frameswreturns.pk") > 0]
weights_files = [x for x in weights_files if x.find("weights.pk") > 0]
                 
for frame_file, weight_file in zip(frames_files, weights_files):
    print(frame_file,":")
    with open(os.path.join(home_dir_frames, frame_file), 'rb') as file:
        frame = pickle.load(file)
    with open(os.path.join(home_dir_weights, weight_file), 'rb') as file:
        weights = pickle.load(file)

    results = calculate_results(weights, frame)

    for KEY in results.keys():
        print(KEY)
        tbhrp_results = np.array(results[KEY])
        print(np.mean(tbhrp_results))
        print(np.std(tbhrp_results))
        print(np.mean(tbhrp_results) / np.std(tbhrp_results))

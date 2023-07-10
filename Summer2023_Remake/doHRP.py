#import re
#import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch,random,numpy as np,pandas as pd
from os import listdir
from os.path import isfile, join
#import math
#import pickle
#from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
#from classhrp import GenerateSIMMAT
simplefilter("ignore", ClusterWarning)

###This part may be deleated in the future
API_KEY = "671c64aa0b622cba50aeaf51f54b8ee209467479d94a16c9e4bfc7badc36abf9"

YEARS = [2016,2020]

rnum = "123"


# On 20151227 by MLdP <lopezdeprado@lbl.gov>
# Hierarchical Risk Parity
#------------------------------------------------------------------------------
def getIVP(cov,**kargs):
    # Compute the inverse-variance portfolio
    ivp=1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp
#------------------------------------------------------------------------------
def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice

    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar
#------------------------------------------------------------------------------
def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3] # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2) # make space
        df0 = sortIx[sortIx >= numItems] # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0] # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0]) # item 2
        sortIx = sortIx.sort_index() # re-sort
        sortIx.index = range(sortIx.shape[0]) # re-index
    return sortIx.tolist()
#------------------------------------------------------------------------------
def getRecBipart(cov,sortIx):
    # Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster
    #print(cItems)
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,k in ((0,len(i)//2), (len(i)//2,len(i))) if len(i)>1]
        #print(cItems)# bi-section
        for i in range(0,len(cItems),2): # parse in pairs
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            #print(cItems0, cItems1)
            cVar0=getClusterVar(cov,cItems0)
            cVar1=getClusterVar(cov,cItems1)
            #print("Cluster Var: ",cVar0, cVar1)
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2
    return w
#------------------------------------------------------------------------------
def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist=((1-corr)/2.)**.5 # distance matrix
    return dist
 
def min_var(cov):
    icov = np.linalg.inv(np.matrix(cov))
    one = np.ones((icov.shape[0],1))
    weights = (icov*one)/(one.T*icov*one)
    return weights

def convert_to_mat(df_):
    keep = GenerateSIMMAT(API_KEY, TICKERS, YEARS)
    keep, unclean = keep.create_simmat()

    mat_ = np.asmatrix(np.split(unclean.values[0],5))[1,0]

import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch

def getFilingURLSin(TICKERS,API_KEY):
    total_response = pd.DataFrame()
    queryApi = QueryApi(api_key=API_KEY)
    for from_batch in range(0, 9800, 200): 
        try:
            payload = {
                "query": {
                    "query_string": {
                        "query": "ticker:({0}) AND formType:\"10-K\"".format(", ".join(map(str, TICKERS)))
                    }
                },
                "from": "{0}".format(from_batch),
                "size": "200", # dont change this
                "sort": [{ "filedAt": { "order": "desc" } }]
            }
            response = queryApi.get_filings(payload)
            print(len(response["filings"]))
            if len(response["filings"]) == 0:
                break
            #print(pd.DataFrame.from_records(response['filings']))
            temp_data = pd.DataFrame.from_records(response['filings'])
            temp_data = temp_data.sort_values(['ticker','filedAt'], ascending = False)
            temp_data['fyear'] = pd.to_datetime(temp_data['periodOfReport'], errors='coerce').dt.year
            #temp_data = temp_data[temp_data.fyear.isin(range(self.YEARS[0]-1,self.YEARS[1]+1))]
            temp_data['filedAt'] = pd.to_datetime(temp_data['filedAt'].str[:10])
            temp_data = temp_data.groupby(["ticker",'fyear']).head(1)
            temp_data = temp_data.reset_index()
            total_response = pd.concat([total_response,temp_data])
        except:
            break
    return total_response



# Read the CSV file
df = pd.read_csv(r"C:\Users\blake\top500Total.csv")
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



# Sample 10 columns randomly and drop rows with NaN values
df = df.sample(30, axis=1).dropna()

# Get the tickers from columns
TICKERS = df.columns.to_list()

# Generate SIMMAT using API_KEY and YEARS
YEARS_TEXT = [YEARS[0]- 1, YEARS[1]]



#keep = GenerateSIMMAT(API_KEY, TICKERS, YEARS_TEXT)
#keep, unclean = keep.create_simmat()

def final_fun(df, TICKERS, keep_unclean):
    save_dict = {}
    # Compute covariance and correlation matrices
    cov, corr = df.cov(), df.corr()
    save_dict['Dates'] = df.index

    ### Traditional HRP
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    hrp = getRecBipart(cov, sortIx)

    # Compute pairwise similarity using TF-IDF
    tfidf = np.asmatrix(np.split(keep_unclean.values[0], len(TICKERS)))
    pairwise_similarity = np.asarray((tfidf * tfidf.T))

    # Adjust pairwise similarity to avoid sqrt error
    pairwise_similarity = ((1 - pairwise_similarity) / 2.)
    np.fill_diagonal(pairwise_similarity, 1)
    pairwise_similarity = np.sqrt(pairwise_similarity)
    np.fill_diagonal(pairwise_similarity, 0)
    dist = pairwise_similarity
    np.nan_to_num(dist, copy=False)

    # Sort and link
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()

    # Capital allocation with text-based sorting
    tbhrp = getRecBipart(cov, sortIx)

    # Print HRP and TB-HRP values
    print("HRP Values:")
    print(hrp.sort_values(ascending=False))
    save_dict['HRP'] = hrp.sort_values(ascending=False)
    print("TB-HRP Values:")
    print(tbhrp.sort_values(ascending=False))
    save_dict['TBHRP'] = tbhrp.sort_values(ascending=False)

    ###Minimum Variance
    print("Minimum Variance:\n",)
    print(pd.Series(np.asarray(min_var(cov).T)[0], cov.columns).sort_values(ascending = False))
    save_dict['MV'] = pd.Series(np.asarray(min_var(cov).T)[0], cov.columns).sort_values(ascending = False)

    #### Inverse Variance
    print("Inverse Variance:\n",)
    iv_weights = df.std().values
    iv_weights = iv_weights / np.linalg.norm(iv_weights, ord = 1)
    print(pd.Series(iv_weights, df.columns).sort_values(ascending = False))
    save_dict['IV'] = pd.Series(iv_weights, df.columns).sort_values(ascending = False)

    return save_dict

#final_fun(df, TICKERS)






'''

#### This is just to plot the dendrogram
plt.style.use("bmh")

ax = plt.gca()
ax.get_yaxis().set_visible(False)
ax.grid(False)
ax.set(frame_on=False)
sch.dendrogram(link, labels = sortIx)
plt.show()
####

'''
# Update the dataframe with log returns and set index
df = return_.copy()
#df.index = date_vector
#keep_list = [x for x in itit__.ticker.tolist() if x in df.columns]
#df = df[keep_list]


# Sample 10 columns randomly and drop rows with NaN values
df = df.sample(30, axis=1).dropna()

# Get the tickers from columns
TICKERS = df.columns.to_list()

# Generate SIMMAT using API_KEY and YEARS
YEARS_TEXT = [YEARS[0]- 1, YEARS[1]]

# Generate SIMMAT using API_KEY and YEARS
keep = GenerateSIMMAT(API_KEY, TICKERS, YEARS_TEXT)
keep, unclean = keep.create_simmat()

df.index = pd.to_datetime([x[0] for x in df.index])
df = df[(df.index >= unclean.index.min()) & (df.index <= unclean.index.max())]
n = 30
list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]



total_save = []
for frame_num in list_df:

    ### Get max date and find the right matrix for the optimization
    max_date = frame_num.index.max()
    keep_unclean = unclean[(unclean.index <= max_date)]
    keep_unclean = keep_unclean[(keep_unclean.index == keep_unclean.index.max())]
    
    total_save.append(final_fun(frame_num, TICKERS, keep_unclean))

'''
######################### Change this ###########################
### This is to intrepret returns
for tpe in ['HRP', 'TBHRP', 'IV', 'EQ','MV']:
    type = tpe  #### 'HRP', 'TBHRP', 'MV', 'IV', 'EQ'
    n = 30
    for x in total_save:
        x['EQ'] = pd.Series([1/n for x in range(n)], index = x['HRP'].index)
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]
    results = []
    for loop in range(len(list_df)-1):
        ### First set in right order
        list_df[loop+1] = list_df[loop+1][total_save[loop][type].index]

        ### Add one
        list_df[loop+1] += 1

        ### Mutiply first row
        list_df[loop+1].iloc[0] *= total_save[loop][type]

        ### Final Return
        results.append(list_df[loop+1].cumprod(axis = 0).iloc[-1].sum()-1)
    #plt.plot(results, label = type)
    #print(type,'\n')
    #print(np.mean(np.array(results)))
    #print(np.std(np.array(results)))
    #print(np.mean(np.array(results))/np.std(np.array(results)))
'''
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

import pickle as pk

with open(rnum + "pandas_frameswreturns.pk",'wb') as file_:
    pk.dump(list_df, file_)
with open(rnum + "weights.pk",'wb') as file_:
    pk.dump(total_save, file_)
### This file is the class that makes the text based groups.

import pandas as pd
import requests as rq
import numpy as np
from sec_api import QueryApi
from sklearn.feature_extraction.text import TfidfVectorizer

#API_KEY = "671c64aa0b622cba50aeaf51f54b8ee209467479d94a16c9e4bfc7badc36abf9"

#TICKERS = ["MSFT","GM","PG","IBM","AA"]
#YEARS = [2012,2022]

class GenerateSIMMAT:
    def __init__(self, API_KEY, TICKERS, YEARS):
        self.API_KEY = API_KEY
        self.TICKERS = TICKERS
        self.YEARS = YEARS
        
    def getFilingURLS(self):
        queryApi = QueryApi(api_key=self.API_KEY)
        payload = {
            "query": {
                "query_string": {
                    "query": "ticker:({0}) AND formType:\"10-K\"".format(", ".join(map(str, self.TICKERS)))
                }
            },
            "sort": [{ "filedAt": { "order": "desc" } }]
        }
        response = queryApi.get_filings(payload)
        temp_data = pd.DataFrame.from_records(response['filings'])
        temp_data = temp_data.sort_values(['ticker','filedAt'], ascending = False)
        temp_data['fyear'] = pd.to_datetime(temp_data['periodOfReport'], errors='coerce').dt.year
        temp_data = temp_data[temp_data.fyear.isin(range(self.YEARS[0]-1,self.YEARS[1]+1))]
        temp_data['filedAt'] = pd.to_datetime(temp_data['filedAt'].str[:10])
        temp_data = temp_data.groupby(["ticker",'fyear']).head(1)
        temp_data = temp_data.reset_index()
        return temp_data
    
    def get_SimilarityMAT(self, FILING_URLS):
        data_ = []
        for url in FILING_URLS:
            payload = {
                "token" : self.API_KEY,
                "item": "1",
                "url": url,
                "type": "text"
            }
            RQ_URL = "https://api.sec-api.io/extractor"
            request = rq.get(RQ_URL, params=payload)
            data_.append(request.text)
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf = vect.fit_transform(data_)
        return np.asarray((tfidf * tfidf.T).todense())
    
    def get_Text_byTICandDate(self):
        temp_data = self.getFilingURLS()
        data_ = {x: {} for x in temp_data.ticker}
        for index, url in temp_data.iterrows():
            payload = {
                "token" : self.API_KEY,
                "item": "1",
                "url": url.linkToFilingDetails,
                "type": "text"
            }
            RQ_URL = "https://api.sec-api.io/extractor"
            request = rq.get(RQ_URL, params=payload)
            data_[url.ticker][url.filedAt] = request.text
        return data_
    
    def make_Timeline(self):
        data_ = self.get_Text_byTICandDate()
        all_ = self.process_data()
        TICKERS = self.TICKERS
        TICKERS.sort()
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        save_results = {}

        for a in all_:
            mat = [data_[tic][a[i]] for i, tic in enumerate(TICKERS)]
            tfidf = vect.fit_transform(mat)
            save_results[max(a)] = np.asarray((tfidf * tfidf.T).todense())

        return save_results
    
    def create_simmat(self):
        ###Primary Function
        save_results = self.make_Timeline()
        TICKERS = self.TICKERS
        TICKERS.sort()
        parsed_data = []

        for timestamp, array in save_results.items():
            flattened_array = np.array(array).flatten()
            parsed_data.append(flattened_array.tolist())
        
        df = pd.DataFrame(parsed_data)
        unclean = df.copy()
        unclean.index = save_results.keys()
        unclean = unclean.sort_index()
        unclean = unclean.replace(to_replace=0, method='ffill')

        df = df.T.drop_duplicates().T
        df.index = save_results.keys()
        df = df.sort_index()
        df = df.replace(to_replace=0, method='bfill')

        #df.index = save_results.keys()
        df = df.loc[:, ~(df.sum(axis=0) >= len(df) - 1)]
        #df.columns = [(TICKERS[y], TICKERS[x]) for y in range(len(TICKERS)) for x in range(y + 1, len(TICKERS))]
        #df = df.sort_index()

        return df, unclean
    
    def find_max_negative_index(self, lst):
        max_index = None
        max_value = float('-inf')  # Initialize with negative infinity

        for i, num in enumerate(lst):
            if num <= 0 and num > max_value:
                max_index = i
                max_value = num

        return max_index

    def process_data(self):
        temp_data = self.getFilingURLS()
        li_ = temp_data.groupby(['ticker'])['filedAt'].apply(lambda x: x.tolist()).tolist()

        idxs = []
        for x in temp_data['filedAt'].unique():
            keep = [self.find_max_negative_index([(z - x).days for z in y]) for y in li_]
            if None not in keep:
                idxs.append(keep)

        idxs = np.array(idxs).T

        all_ = [[li_[ip][i] for i in r] for ip, r in enumerate(idxs)]

        return np.array(all_).T

### This file is the class that makes the text based groups.

import pandas as pd
import requests as rq
import numpy as np
from sec_api import QueryApi
from sklearn.feature_extraction.text import TfidfVectorizer


#TICKERS = ["MSFT","GM","PG","IBM","AA"]
#YEARS = [2012,2022]

class GenerateSIMMAT:
    def __init__(self, API_KEY, TICKERS, YEARS):
        self.API_KEY = API_KEY
        self.TICKERS = TICKERS
        self.YEARS = YEARS
        self.queryApi = QueryApi(api_key=self.API_KEY)
        self.ticker_str = ", ".join(map(str, self.TICKERS))
        
    def getFilingURLS(self):
        total_response = pd.DataFrame()
        for from_batch in range(0, 9800, 200): 
            payload = {
                "query": {
                    "query_string": {
                        "query": f"ticker:({self.ticker_str}) AND formType:\"10-K\" AND filedAt:[{str(self.YEARS[0])}-01-01 TO {str(self.YEARS[1])}-12-31]"
                    }
                },
                "from": f"{from_batch}",
                "size": "200", # dont change this
                "sort": [{ "filedAt": { "order": "desc" } }]
            }
            response = self.queryApi.get_filings(payload)
            if not response["filings"]:
                break
            temp_data = pd.DataFrame(response['filings'])
            temp_data = temp_data.sort_values(['ticker','filedAt'], ascending=False)
            temp_data['fyear'] = pd.to_datetime(temp_data['periodOfReport'], errors='coerce').dt.year
            temp_data['filedAt'] = pd.to_datetime(temp_data['filedAt'].str[:10])
            temp_data = temp_data.groupby(["ticker",'fyear']).head(1)
            total_response = pd.concat([total_response,temp_data])
        return total_response.reset_index()
    
    def get_SimilarityMAT(self, FILING_URLS):
        data_ = [rq.get("https://api.sec-api.io/extractor", params={
                "token" : self.API_KEY,
                "item": "1",
                "url": url,
                "type": "text"}).text for url in FILING_URLS]
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf = vect.fit_transform(data_)
        return (tfidf * tfidf.T).A #converts to numpy array, which is more efficient than todense()
    
    def get_Text_byTICandDate(self):
        temp_data = self.getFilingURLS()
        data_ = {x: {} for x in temp_data.ticker}
        for _, row in temp_data.iterrows():
            data_[row.ticker][row.filedAt] = rq.get(
                "https://api.sec-api.io/extractor",
                params={
                    "token" : self.API_KEY,
                    "item": "1",
                    "url": row.linkToFilingDetails,
                    "type": "text"
                }).text
        return data_
    
    def make_Timeline(self):
        data_ = self.get_Text_byTICandDate()
        all_ = self.process_data()
        self.TICKERS.sort()
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        save_results = {max(a): (vect.fit_transform([data_[tic][a[i]] for i, tic in enumerate(self.TICKERS)]) * vect.fit_transform([data_[tic][a[i]] for i, tic in enumerate(self.TICKERS)]).T).A
                        for a in all_}
        return save_results
    
    def create_simmat(self):
        save_results = self.make_Timeline()
        self.TICKERS.sort()
        parsed_data = [np.array(array).flatten().tolist() for array in save_results.values()]
        
        df = pd.DataFrame(parsed_data)
        unclean = df.copy()
        unclean.index = save_results.keys()
        unclean.sort_index(inplace=True)
        unclean.replace(to_replace=0, method='ffill', inplace=True)

        df = df.T.drop_duplicates().T
        df.index = save_results.keys()
        df.sort_index(inplace=True)
        df.replace(to_replace=0, method='bfill', inplace=True)

        df = df.loc[:, ~(df.sum(axis=0) >= len(df) - 1)]

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
        li_ = temp_data.groupby(['ticker'])['filedAt'].apply(list).tolist()

        idxs = [[self.find_max_negative_index([(z - x).days for z in y]) for y in li_] for x in temp_data['filedAt'].unique() if None not in [self.find_max_negative_index([(z - x).days for z in y]) for y in li_]]

        all_ = [[li_[ip][i] for i in r] for ip, r in enumerate(np.array(idxs).T)]

        return np.array(all_).T

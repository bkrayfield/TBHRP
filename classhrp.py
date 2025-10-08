### This file is the class that makes the text based groups.

import pandas as pd
import requests as rq
import numpy as np
from sec_api import QueryApi
from sklearn.feature_extraction.text import TfidfVectorizer

class GenerateSIMMAT:
    """
    Generates similarity matrices from 10-K filings.
    Can operate in online mode (fetching data via SEC-API) or using preloaded data.
    """
    def __init__(self, TICKERS, YEARS, api_key=None, preloaded_filing_urls=None, preloaded_filing_texts=None):
        """
        Initializes the class.

        Args:
            TICKERS (list): A list of company tickers.
            YEARS (list): A list with the start and end year.
            api_key (str, optional): API key for online mode.
            preloaded_filing_urls (pd.DataFrame, optional): An Optional URL File for using non SEC-API data.
            preloaded_filing_texts (dict, optional): An Optional Text File for using non SEC-API data.
        """
        # Validate the configuration for online vs. preloaded mode
        is_offline = preloaded_filing_urls is not None and preloaded_filing_texts is not None
        if not is_offline and not api_key:
            raise ValueError("An SEC-API key is required for online mode. For preloaded data, both 'preloaded_filing_urls' and 'preloaded_filing_texts' must be provided. An example file is included in the repository.")

        self.API_KEY = api_key
        self.TICKERS = TICKERS
        self.YEARS = YEARS
        
        # Store preloaded data for preloaded mode
        self._preloaded_urls = preloaded_filing_urls
        self._preloaded_texts = preloaded_filing_texts

        # Initialize API objects only if in online mode
        if self.API_KEY:
            self.queryApi = QueryApi(api_key=self.API_KEY)
            self.ticker_str = ", ".join(map(str, self.TICKERS))
        
    def getFilingURLS(self):
        """
        Returns a DataFrame of filing metadata. In preloaded mode, it returns
        the preloaded data. In preloaded mode, it fetches data from the API.
        """
        if self._preloaded_urls is not None:
            print("  Using preloaded filing URLs.")
            return self._preloaded_urls
        
        # Original online mode logic
        total_response = pd.DataFrame()
        for from_batch in range(0, 9800, 200): 
            payload = {
                "query": { "query_string": {
                        "query": f"ticker:({self.ticker_str}) AND formType:\"10-K\" AND filedAt:[{self.YEARS[0]}-01-01 TO {self.YEARS[1]}-12-31]"
                }},
                "from": f"{from_batch}", "size": "200",
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
    
    def get_Text_byTICandDate(self):
        """
        Returns a dictionary of filing texts. In preloaded mode, it returns
        the preloaded data. In online mode, it fetches data from the API.
        """
        if self._preloaded_texts is not None:
            print(" Using preloaded filing texts.")
            return self._preloaded_texts

        # Original online mode logic
        temp_data = self.getFilingURLS()
        data_ = {x: {} for x in temp_data.ticker}
        for _, row in temp_data.iterrows():
            data_[row.ticker][row.filedAt] = rq.get(
                "https://api.sec-api.io/extractor",
                params={
                    "token": self.API_KEY, "item": "1",
                    "url": row.linkToFilingDetails, "type": "text"
                }).text
        return data_
    
    def get_SimilarityMAT(self, FILING_URLS):
        data_ = [rq.get("https://api.sec-api.io/extractor", params={
                "token" : self.API_KEY, "item": "1",
                "url": url, "type": "text"}).text for url in FILING_URLS]
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf = vect.fit_transform(data_)
        return (tfidf * tfidf.T).A

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
        max_value = float('-inf')
        for i, num in enumerate(lst):
            if num <= 0 and num > max_value:
                max_index, max_value = i, num
        return max_index

    def process_data(self):
        temp_data = self.getFilingURLS()
        li_ = temp_data.groupby(['ticker'])['filedAt'].apply(list).tolist()
        idxs = [[self.find_max_negative_index([(z - x).days for z in y]) for y in li_] for x in temp_data['filedAt'].unique() if None not in [self.find_max_negative_index([(z - x).days for z in y]) for y in li_]]
        all_ = [[li_[ip][i] for i in r] for ip, r in enumerate(np.array(idxs).T)]
        return np.array(all_).T
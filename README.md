# TBHRP -  Hierarchical Risk Parity with Text-Based Clustering

This project implements the Hierarchical Risk Parity (HRP) algorithm, an alternative portfolio optimization method that does not require the inversion of the covariance matrix, making it more robust to estimation errors. This implementation extends the traditional HRP by incorporating text-based clustering of assets using TF-IDF (Term Frequency-Inverse Document Frequency) from financial filings (10-K reports), allowing for a "Text-Based HRP" (TB-HRP) that groups similar companies based on their textual descriptions.

## Files

### `classhrp.py`
This file defines the `GenerateSIMMAT` class, which is responsible for fetching financial filing data and generating similarity matrices based on the textual content of these filings.

**Key Features:**
- Fetches 10-K filings from the SEC API for a given list of tickers and years.
- Extracts Item 1 (Business Description) text from the 10-K filings.
- Uses TF-IDF to vectorize the text data.
- Computes a similarity matrix between companies based on their TF-IDF vectors.
- Provides functionality to create a timeline of similarity matrices, allowing for analysis over different periods.

### `doHRP.py`
This script orchestrates the HRP and TB-HRP portfolio allocation processes. It integrates the `GenerateSIMMAT` class to obtain text-based similarity and then applies the HRP algorithm. It also includes functions for traditional portfolio optimization methods like Minimum Variance and Inverse Variance.

**Key Features:**
- Calculates log returns from historical stock price data.
- Implements the core HRP algorithm, including:
    - `getIVP`: Computes the inverse-variance portfolio.
    - `getClusterVar`: Calculates the variance per cluster.
    - `getQuasiDiag`: Sorts clustered items by distance.
    - `getRecBipart`: Computes HRP allocation by recursively bisecting.
- Implements `correlDist` for correlation-based distance.
- `min_var` for Minimum Variance portfolio weights.
- Generates and applies text-based similarity for TB-HRP.
- Compares HRP, TB-HRP, Minimum Variance, and Inverse Variance portfolio allocations.
- Saves the generated pandas dataframes and calculated weights to pickle files.

### `results_.py`
This script is designed to load the saved portfolio weights and pandas dataframes, and then calculate and print the performance metrics for each portfolio optimization strategy (HRP, TB-HRP, Inverse Variance, Equal Weight, and Minimum Variance).

**Key Features:**
- Loads pickled pandas dataframes (containing returns) and portfolio weights.
- Calculates the returns for each portfolio strategy over different periods.
- Prints the mean, standard deviation, and Sharpe ratio (mean/std) for the TB-HRP strategy.

## Getting Started

### Prerequisites
- Python 3.x
- `pandas`
- `requests`
- `numpy`
- `scikit-learn`
- `scipy`
- `sec_api` (Requires an API key from sec-api.io)
- `matplotlib` (Optional, for plotting dendrograms)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name



### API Key

You will need an API key from sec-api.io. Once you have your API key, update the API_KEY variable in doHRP.py and classhrp.py.


### Data

The doHRP.py script expects a CSV file named top500Total.csv containing historical stock price data. The first column should be an 'idx' representing dates, and subsequent columns should be ticker symbols with their corresponding prices. You will need to provide this file or modify the df = pd.read_csv(...) line in doHRP.py to point to your data.

### Running the Scripts

1. Generate Similarity Matrices and Weights:
    Execute doHRP.py to fetch filing data, compute similarity matrices, and calculate portfolio weights for different strategies. This script will save the results as pickle files (<rnum>pandas_frameswreturns.pk and <rnum>weights.pk, where rnum is a random number generated in the script).

2. Analyze Results:
    Execute results_.py to load the saved results and calculate performance metrics. Make sure to adjust the home_dir_frames and home_dir_weights variables in results_.py to the directory where your pickle files are saved.

Project Structure
```bash
    .
    ├── classhrp.py
    ├── doHRP.py
    ├── results_.py
    ├── top500Total.csv (example data file - not included in repo, user must provide)
    └── README.md

Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for any bugs, feature requests, or improvements.
License

This project is open-source and available under the MIT License.